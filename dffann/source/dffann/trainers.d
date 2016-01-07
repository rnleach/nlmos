/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: May 11, 2015
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
import dffann.errorFunctions;

version(unittest)
{
  import std.math;

  import dffann.dffann;
  import dffann.testUtilities.testData;
  import dffann.linearNetworks;
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
  public @property feedforwardnetwork net();
}


/**
 *
 */
public abstract class AbstractTrainer(size_t nInputs, size_t nTargets): Trainer
{
  alias iData = immutable(Data!(nInputs,nTargets));

  protected double _error = double.max;
  protected feedforwardnetwork _net = null;
  protected iData _tData;

  /**
   * Params:
   * inNet        = A network that will serve as a template, it's not trained
   *                in place.
   * trainingData = The data used to train the network.
   */
  public this(feedforwardnetwork inNet, iData trainingData)
  {
    this._net = inNet.dup;
    this._tData = trainingData;
  }

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
  override @property feedforwardnetwork net() { return this._net.dup; }
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
  public this(feedforwardnetwork inNet, iData trainingData)
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
    Matrix beta = design.T * targets;
    SVDDecomp svd = SVDDecomp(alpha);
    Matrix inverseAlpha = svd.pseudoInverse;
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
    auto ef = new ErrorFunction!(EFType.ChiSquare, typeof(_tData))(_net, _tData);
    ef.evaluate(p, false);
    this._error = ef.value;
  }
}

unittest
{
  mixin(dffann.dffann.announceTest("LinearTrainer"));
  
  // Constants to define test data
  enum numPoints = 1500;
  enum numIn = 6;
  enum numOut = 2;

  // short hand for dealing with data
  alias Data!(numIn, numOut) DataType;
  alias iData = immutable(Data!(numIn, numOut));
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
  feedforwardnetwork trainedNet = lt.net;

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
public class BFGSTrainer(size_t nInputs, size_t nTargets, EFType): 
AbstractTrainer!(nInputs, nTargets)
{

  // Parameters for tuning the optimization
  /**
   *
   */
  public size_t numBatches = 1;

  /**
   *
   */
  public size_t maxIt = 1_000;

  /**
   *
   */
  public double minDeltaE = 2.0 * sqrt(double.min) + double.min;

  // -------------------------------------------------------------------------
  

  /**
   * Params:
   * inNet        = A network that will serve as a template, it's not trained
   *                in place.
   * trainingData = The data used to train the network.
   */
  public this(feedforwardnetwork inNet, iData trainingData) 
  {
    super(inNet, trainingData);

  }

  /**
   * See_Also:
   * Trainer.train
   */
  override public void train()
  {

    /+ Linear Trainer Code
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
    Matrix beta = design.T * targets;
    SVDDecomp svd = SVDDecomp(alpha);
    Matrix inverseAlpha = svd.pseudoInverse;
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
    auto ef = new ErrorFunction!(EFType.ChiSquare, typeof(_tData))(_net, _tData);
    ef.evaluate(p, false);
    this._error = ef.value;
    +/
  }

}
