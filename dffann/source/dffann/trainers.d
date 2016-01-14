/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: January 13, 2016
 *
 * This module contains classes for training networks.
 *
 */
module dffann.trainers;

import numeric.func;
import numeric.matrix;
import numeric.minimize;

import dffann.data;
import dffann.feedforwardnetwork;
import dffann.errorfunctions;

version(unittest)
{
  import std.math;
  import std.stdio;
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
private abstract class AbstractTrainer(size_t nInputs, size_t nTargets): Trainer
{
  alias iData = immutable(Data!(nInputs,nTargets));

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
public class LinearTrainer(size_t nInputs, size_t nTargets): AbstractTrainer!(nInputs, nTargets)
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
    auto nPoints = _tData.nPoints;
    auto nPredictors = _tData.nInputs + 1; // +1 for bias
    auto nPredictands = _tData.nTargets;

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
    auto ef = new ErrorFunction!(EFType.ChiSquare, typeof(_tData.simpleRange))(_net, _tData.simpleRange);
    ef.evaluate(p, false);
    this._error = ef.value;
  }
}

///
unittest
{
  // Constants to define test data
  enum numPoints = 1500;
  enum numIn = 6;
  enum numOut = 2;

  // short hand for dealing with data
  alias DataType = Data!(numIn, numOut);
  alias iData = immutable(DataType);
  alias DP = immutable(DataPoint!(numIn, numOut));

  // Generate some raw test data and binary flags to match it.
  double[][] testData = makeLinearRegressionData(numPoints, numIn, numOut, 0.0);
  bool[] binFlags = [false];
  foreach(i; 1 .. (numIn + numOut)){ binFlags ~= false; }
  
  // Make a data set
  iData d1 = DataType.createImmutableData(testData, binFlags);


  // Make a trainer, and supply it with a network to train.
  LinearTrainer!(numIn, numOut) lt = 
            new LinearTrainer!(numIn, numOut)(new LinRegNet(numIn,numOut), d1);

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
 * TODO
 * 
 * TODO - consider adding contract checks to ensure error function types and
 *        output activation functions are properly matched.
 */
public class BFGSTrainer(size_t nInputs, size_t nTargets, EFType erf, bool randomOrder = false): 
AbstractTrainer!(nInputs, nTargets)
{

  // Parameters for tuning the optimization

  /// The maximum number of iterations to attempt.
  public size_t maxIt = 1_000;

  /// Maximum number of time to try.
  public uint maxTries = 2;

  /// A stopping criterion for changes in the error function.
  public double minDeltaE = 2.0 * sqrt(double.min_normal) + double.min_normal;

  /**
   * A regulizer to use while training. May be null, defualts to null.
   *
   * See_Also: dffan.errorfunctions.Regulizer
   */
  public Regulizer regulizer = null;

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

    static if(randomOrder)
    {
      auto dataRange = _tData.randomRange;
    }
    else
    {
      auto dataRange = _tData.simpleRange;
    }

    // Make an error function
    auto ef = new ErrorFunction!(erf, typeof(dataRange))(_net, dataRange, regulizer);

    // Try several times
    double[] bestParms = _net.parameters.dup;
    foreach(uint trie; 0 .. maxTries)
    {
      double[] parms = _net.parameters.dup;
      version(unittest) writefln("try %d and parms: %s", trie, parms);

      try
      {
        // Minimize
        bfgsMinimize(ef, parms, maxIt, minDeltaE);
      }
      catch(FailureToConverge fc)
      {
        // Ignore it, we may be in a better place than we started, even if it
        // is not good.
        //assert(0);
        version(unittest)
        {
          writef("Failed to converge...");
        }
      }

      // Evaluate the error one more time and set it to this error
      ef.evaluate(parms, false);
      version(unittest) writefln("try %d done and error = %s parms: %s", trie, ef.value, parms);

      // Only accept these parameters if they improve the error.
      if( ef.value < _error)
      {
        version(unittest)
        {
          writefln("old error: %s, new error: %s, parms: %s", _error, ef.value, parms);
        }
        _error = ef.value;
        // Remember the best parameters so far.
        bestParms = parms.dup;
      }
      else
      {
        version(unittest)
        {
          writeln();
        }
      }

      // Set random parameters and try again.
      _net.setRandom();
    }
    _net.parameters = bestParms;
  }
}

///
unittest
{
  // Constants to define test data
  enum numIn = 2;
  enum numOut = 1;
  const uint[] numNodes = [numIn, 2, numOut];

  // short hand for dealing with data
  alias DataType = Data!(numIn, numOut);
  alias iData = immutable(DataType);
  alias DP = immutable(DataPoint!(numIn, numOut));

  // Generate some raw test data and binary flags to match it.
  double[][] testData = [[0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0]];
  bool[] binFlags = [true];
  foreach(i; 1 .. (numIn + numOut)){ binFlags ~= true; }
  
  // Make a data set
  iData d1 = DataType.createImmutableData(testData, binFlags);


  // Make a trainer, and supply it with a network to train.
  auto net = new MLP2ClsNet(numNodes);
  net.parameters = [ 50.0, -50.0, -25, 
                    -50.0,  50.0, -25, 
                     50.0,  50.0, 25];
  auto bfgs_t = 
            new BFGSTrainer!(numIn, numOut, EFType.CrossEntropy2C)(net, d1);
  bfgs_t.minDeltaE = 1.0e-20;
  bfgs_t.maxIt = 1_000_000;
  bfgs_t.maxTries = 5;

  // Train the network and retrieve the newly trained network.
  bfgs_t.train;
  FeedForwardNetwork trainedNet = bfgs_t.net;

  // Since we supplied data with no noise added, it should be a perfect fit,
  // so the error should be zero!
  writefln("Error = %s ", bfgs_t.error);
  //assert(approxEqual(bfgs_t.error,0.0), format("Error is %s.", bfgs_t.error));
  // The network should perfectly map the inputs to the targets.
  foreach(dp; d1.simpleRange)
  {
    //assert(approxEqual(trainedNet.eval(dp.inputs),dp.targets));
    writefln("Inputs: %5s    Evaluated: %5s   Targets: %5s", dp.inputs, trainedNet.eval(dp.inputs), dp.targets);
  }
}
