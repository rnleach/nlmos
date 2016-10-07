/**
* Author: Ryan Leach
* Version: 1.0.0
* Date: January 13, 2016
*
* This module contains classes for training networks.
*
*/
module dffann.trainers;

import std.math;

import numeric;
import numeric.func;
import numeric.matrix;
import numeric.minimize;

import dffann;
import dffann.data;
import dffann.feedforwardnetwork;
import dffann.errorfunctions;
import dffann.strategy;

version(unittest)
{
  import std.stdio;
  import std.algorithm;
  import dffann.testutilities.testdata;
  import dffann.linearnetworks;
  import dffann.multilayerperceptrons;
}

/**
* A simple interface for trainers so they can be switched around easily or
* composed into higher level trainers that use trainers.
*/
public interface Trainer
{
  /**
  * Perform the training, this is often a resource intensive process as it
  * involves minimizing an error function.
  */
  public void train();

  /**
  * Get the error, this method assumes the train method has already been 
  * called.
  */
  public @property double error();

  /**
  * Get a copy of the network that was trained, again, this method assumes
  * the train method was already called.
  */
  public @property FeedForwardNetwork net();
}

/**
* Common functionality for Linear and BFGS trainers. Others may be added later.
*/
private abstract class AbstractTrainer: Trainer
{
  alias iData = immutable(Data);

  protected double _error = double.max;
  protected FeedForwardNetwork _net = null;
  protected iData _tData;

  /**
  * Params:
  * inNet        = A network that will serve as a template, it's not trained
  *                in place.
  * trainingData = The data used to train the network.
  */
  public this(FeedForwardNetwork inNet, iData trainingData)
  {
    assert( inNet.numInputs  == trainingData.nInputs  );
    assert( inNet.numOutputs == trainingData.nTargets );

    this._net = inNet.dup;
    this._tData = trainingData;
  }

  /**
  * See_Also:
  * Trainer.train
  */
  override public abstract void train();

  /**
  * See_Also:
  * Trainer.error
  */
  override @property double error(){ return this._error; }

  /**
  * See_Also:
  * Trainer.net
  */
  override @property FeedForwardNetwork net() { return this._net.dup; }
}

/**
* Only trains linear networks.
*/
public class LinearTrainer: AbstractTrainer
{
  
  /**
  * Params:
  * inNet        = A network that will serve as a template, it's not trained
  *                in place.
  * trainingData = The data used to train the network.
  */
  public this(FeedForwardNetwork inNet, iData trainingData)
  {
    super(inNet, trainingData);
  }

  /**
  * See_Also:
  * Trainer.train
  */
  override public void train()
  {

    // Get a few useful constants
    const nPoints = _tData.nPoints;
    const nPredictors = _tData.nInputs + 1; // +1 for bias
    const nPredictands = _tData.nTargets;

    // Make the design matrix and the targets
    Matrix design = Matrix(nPoints, nPredictors);
    Matrix targets = Matrix(nPoints, nPredictands);
    auto dr = _tData.simpleRange;
    
    size_t cnt = 0;
    foreach(dp; dr)
    {
      assert(cnt < design.numRows);

      foreach(i, val; dp.targets)
      {
        targets[cnt,i] = val;
      }
      foreach(i, val; dp.inputs)
      {
        design[cnt,i] = val;
      }
      design[cnt, nPredictors - 1] = 1.0; // Bias

      ++cnt;
    }

    // Check to make sure it worked out
    assert(cnt == nPoints);

    // Now solve with SVD
    Matrix alpha = design.T * design;
    const Matrix beta = design.T * targets;
    const SVDDecomp svd = SVDDecomp(alpha);
    const Matrix inverseAlpha = svd.pseudoInverse;
    Matrix solution = inverseAlpha * beta;

    // Now unpack solution and put it into the linear network parameters
    double[] p = new double[nPredictands * nPredictors];
    size_t j = 0;
    foreach(o; 0 .. nPredictands)
    {
      foreach(i; 0 .. (nPredictors - 1))
      {
        p[j++] = solution[i, o];
      }
      p[j++] = solution[nPredictors - 1, o];
    }

    // Set the parameters in the net
    _net.parameters = p;

    // Calculate the error.
    auto ef = new ErrorFunction!(EFType.ChiSquare)(_net, _tData);
    ef.evaluate(p, false);
    this._error = ef.value;
  }
}

unittest
{
  mixin(announceTest("LinearTrainer"));
  
  // Constants to define test data
  enum numPoints = 1500;
  enum numIn = 6;
  enum numOut = 2;

  // short hand for dealing with data
  alias iData = immutable(Data);

  // Generate some raw test data for regression.
  double[][] testData = makeLinearRegressionData(numPoints, numIn, numOut, 0.0);
  
  // Make a data set
  iData d1 = new immutable(Data)(numIn, numOut, testData);

  // Make a trainer, and supply it with a network to train.
  LinearTrainer lt = new LinearTrainer(new LinRegNet(numIn,numOut), d1);

  // Train the network and retrieve the newly trained network.
  lt.train;
  FeedForwardNetwork trainedNet = lt.net;

  // Since we supplied data with no noise added, it should be a perfect fit,
  // so the error should be zero!
  assert(approxEqual(lt.error,0.0));
  // The network should perfectly map the inputs to the targets.
  foreach(dp; d1.simpleRange)
  {
    assert(approxEqual(trainedNet.eval(dp.inputs),dp.targets));
  }
}

/**
* BFGS Minimization trainer.
*
* Params:
* erf       = The specific error function type to use.
* para      = ParallelStrategy.parallel if you want to use the parallel 
*             (concurrent) code in the evaluation of the error function.
* batchMode = BatchStrategy.mini-batch if you want to use mini-batches, or a 
*             subset of the data for each iteration. Otherwise the default is
*             BatchStrategy.batch to use the whole data set.
* randomize = RandomStrategy.inOrder if you want to go through all the points
*             in a data set in order. This is ignored (defaults to inOrder)
*             UNLESS mini-batches are used. Then the extra effort is worth it 
*             to default to using a random strategy to prevent odd cycles in 
*             the data.
* 
* TODO - consider adding contract checks to ensure error function types and
*        output activation functions are properly matched.
*/
public class BFGSTrainer(EFType erf,
                         ParallelStrategy para   = ParallelStrategy.parallel,
                         BatchStrategy batchMode = BatchStrategy.batch, 
                         RandomStrategy random   = RandomStrategy.random
              ): AbstractTrainer
{

  enum par = para == ParallelStrategy.parallel;
  enum useMinibatches = batchMode == BatchStrategy.minibatch;
  enum randomize = random == RandomStrategy.random;

  // Parameters for tuning the optimization

  /// The maximum number of iterations to attempt.
  public size_t maxIt = 1_000;

  /// Maximum number of times to try.
  public uint maxTries = 1;

  /// A stopping criterion for changes in the error function.
  public double minDeltaE = 2.0 * sqrt(double.min_normal) + double.min_normal;

  static if(useMinibatches)
  {
    /// A batch size to use when evaluating the error with mini-batches.
    public uint batchSize = 10;
  }

  /**
  * A regularizer to use while training. May be null, defaults to null.
  *
  * See_Also: dffan.errorfunctions.Regularizer
  */
  public Regularizer regularizer = null;

  /**
  * Params:
  * inNet        = A network that will serve as a template, it's not trained
  *                in place.
  * trainingData = The data used to train the network.
  *
  */
  public this(FeedForwardNetwork inNet, iData trainingData) 
  {
    super(inNet, trainingData);
  }

  /**
  * See_Also:
  * Trainer.train
  */
  override public void train()
  {
    version(unittest)
    {
      uint failCount = 0;
    }

    // Make an error function
    alias ErrFunType = ErrorFunction!(erf, para, batchMode, random);
    auto ef = new ErrFunType(_net, _tData, regularizer);
    

    static if(useMinibatches) 
    {
      // Set the batch size.
      ef.batchSize = batchSize;

      // Keep this around to evaluate the total error after the minimization,
      // which only uses mini-batches.
      alias TotalErrFun = ErrorFunction!(erf, para, BatchStrategy.batch);
      auto totalErrFun = new TotalErrFun(_net, _tData, regularizer);
    }

    // Try several times
    double[] bestParms = _net.parameters.dup;
    double[] parms = _net.parameters.dup;
    foreach(uint trie; 0 .. maxTries)
    {
      parms[] = _net.parameters[];
      //version(unittest) writefln("try %d and parms: %s", trie, parms);

      try
      {
        // Minimize
        bfgsMinimize(ef, parms, maxIt, minDeltaE);
      }
      catch(FailureToConverge fc)
      {
        // Ignore this error, the minimization did everything it could to get
        // the best result, just live with what we got. It should have left
        // parms at the best value it has achieved so far.
        
        //version(unittest)
        //{
        //  writefln("%d Failed to converge...iterations = %d, tolerance = %g, " ~
        //    "best error = %g.", ++failCount, fc.iterations, fc.tolerance, 
        //    fc.minSoFar);
        //}
        
      }

      // Evaluate the error one more time and set it to this error
      double finalTryError;

      static if(useMinibatches)
      {
        totalErrFun.evaluate(parms, false);
        finalTryError = totalErrFun.value;
      }
      else
      {
        ef.evaluate(parms, false);
        finalTryError = ef.value;
      }
      //version(unittest) writefln("   try %d done and error = %f parms: %s", 
      //  trie, finalTryError, parms);

      // Only accept these parameters if they improve the error.
      if( finalTryError < _error)
      {
        _error = finalTryError;

        // Remember the best parameters so far.
        bestParms[] = parms[];
      }

      // Set random parameters and try again.
      _net.setRandom();
    }
    _net.parameters = bestParms;
  }
}

unittest
{
  mixin(announceTest("BFGSTrainer"));

  // Constants to define test data
  enum numIn = 2;
  enum numOut = 1;
  const uint[] numNodes = [numIn, 2, numOut];

  // short hand for dealing with data
  alias iData = immutable(Data);

  // Generate some raw test data and binary flags to match it.
  double[][] testData = [[0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0]];
  
  // Make a data set
  iData d1 = new iData(numIn, numOut, testData);

  // Make a trainer, and supply it with a network to train.
  // This is a tough network to train, exhibits some pathological behavior by
  // trying to explode the weights to large values. Also has lots of local 
  // minima, so try lots of times.
  auto net = new MLPTanh2ClsNet(numNodes);

  auto bfgs_t = new BFGSTrainer!(EFType.CrossEntropy2C,  
                                 ParallelStrategy.serial)(net, d1);
  bfgs_t.minDeltaE = 1.0e-6;
  bfgs_t.maxIt = 10_000;
  bfgs_t.maxTries = 100; // Should be more than enough attempts

  // Train the network and retrieve the newly trained network.
  bfgs_t.train;
  FeedForwardNetwork trainedNet = bfgs_t.net;

  // This is probability, and not perfect, but it should be close.
  //writefln("Error = %s ", bfgs_t.error);
  assert(approxEqual(bfgs_t.error, 0.0, 1.0e-2, 1.0e-2), 
    format("Error is %s. IF IT FAILS, RUN TEST AGAIN. "~
      "SMALL CHANCE IT WILL FAIL.", bfgs_t.error));

  // The network should perfectly map the inputs to the targets.
  foreach(dp; d1.simpleRange)
  {
    assert(approxEqual(trainedNet.eval(dp.inputs),dp.targets, 1.0e-1, 1.0e-1));
    //writefln("Inputs: %5s    Evaluated: %5s   Targets: %5s", dp.inputs, 
    //  trainedNet.eval(dp.inputs), dp.targets);
  }

  // Try again, this time with random batches of 2.
  enum inNum = 5;
  enum outNum = 2;

  auto net2 = new LinClsNet(inNum, outNum);
  iData d2 = new iData(inNum, outNum, 
    makeLinearClassificationData(2000, inNum, outNum, 0.0));

  auto bfgs_tbr = new BFGSTrainer!(EFType.CrossEntropy)(net2, d2);

  bfgs_tbr.minDeltaE = 1.0e-6;
  bfgs_tbr.maxIt = 1_000;
  bfgs_tbr.maxTries = 1;
  
  // Train the network and retrieve the newly trained network.
  bfgs_tbr.train;
  trainedNet = bfgs_tbr.net;

  // This is probability, and not perfect, but it should be close.
  //writefln("Error = %s ", bfgs_tbr.error);
  assert(approxEqual(bfgs_tbr.error, 0.0, 1.0e-1, 1.0e-1), 
    format("Error is %s. IF IT FAILS, RUN TEST AGAIN. "~
      "SOME CHANCE IT WILL LEGITIMATELY FAIL.", bfgs_tbr.error));

  // The network should perfectly map the inputs to the targets.
  foreach(dp; d2.simpleRange)
  {
    assert(approxEqual(trainedNet.eval(dp.inputs),dp.targets, 1.0e-1, 1.0e-1));
    //writefln("Inputs: %5s    Evaluated: %5s   Targets: %5s", dp.inputs, 
    //  trainedNet.eval(dp.inputs).map!(a => round(a)), dp.targets);
  }
}
