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

public import dffann.dffann;
import dffann.data;
import dffann.feedforwardnetwork;
import dffann.errorFunctions;

/**
 * TODO document interface
 */
public interface Trainer
{
  public void train();
  public @property double error();
  public @property feedforwardnetwork net();
}

/**
 * TODO document class
 */
public class LinearTrainer(size_t nInputs, size_t nTargets): Trainer
{
  alias iData = immutable(Data!(nInputs,nTargets));

  private double _error = double.max;
  private feedforwardnetwork _net = null;
  private iData _tData;

  public this(feedforwardnetwork inNet, iData trainingData)
  {
    this._net = inNet.dup;
    this._tData = trainingData;
  }

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
    while(cnt < design.numRows && !dr.empty)
    {
      auto dp = dr.front;
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
      dr.popFront();
    }

    // Check to make sure it worked out
    assert(cnt == nPoints);

    // Now solve with SVD
    Matrix alpha = design.Tv * design;
    Matrix beta = design.Tv * targets;
    SVDDecomp svd = SVDDecomp(alpha);
    Matrix inverseAlpha = svd.pseudoInverse;
    Matrix solution = inverseAlpha * beta;

    // Now unpack solution and put it inot the linear network parameters
    double[] p = new double[nPredictands * nPredictors];
    size_t j = 0;
    foreach(o; 0 .. nPredictands)
    {
      foreach(i; 0 .. (nPredictors - 1))
      {
        p[j++] = solution[i, j];
      }
      p[j++] = solution[nPredictors - 1, o];
    }
    

    // Set the parameters in the net
    //_net.parameters = p; // Done in ef.evaulate code below

    // Calculate the error.
    auto ef = new ErrorFunction!(EFType.ChiSquare, _tData)(_net, _tData);
    ef.evaluate(p, false);
    this._error = ef.value;

  }

  override @property double error()
  {
    return this._error;
  }

  override @property feedforwardnetwork net()
  {
    return this._net.dup;
  }
}
unittest
{
  mixin(dffann.dffann.announceTest("LinearTrainer"));


}