/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: February 7, 2015
 *
 * This module contains error functions AND regulizers. Regulizers are treated
 * as an addition to the error function.
 *
 */
module dffann.errorFunctions;

import std.array;
import std.math;

import numeric.func;

public import dffann.dffann;
import dffann.data;
import dffann.feedforwardnetwork;

/**
 * Different types of error functions are used to parameterize the error 
 * function class.
 *
 * ChiSquare is the standard sum of square errors averaged over the number
 * of data points in a sample.
 *
 * CrossEntropy is the average per point cross entropy error for multiple 
 * classes in 1-of-N encoding for classification problems.
 *
 * CrossEntropy2C is the average per point cross entropy error for 0-1 encoding
 * of 2 class classification problems where the network can only have a single
 * output.
 */
enum EFType {ChiSquare, CrossEntropy, CrossEntropy2C};

/**
 * A class for computing the value and gradient of an error function for 
 * a feed forward network with a given set of weights. This class implements 
 * the numeric.func.func interface so it can be used with the function 
 * minimization tools in the numeric package during training.
 *
 * Params:
 * errFuncType = The specific error function to calculate.
 * T           = An instantiation of a DataType template as a type.
 * par         = true if you want to use the parallel (concurrent) code in the
 *               evaluate method. You may want to use false if you plan to 
 *               parallelize at a higher level construct.
 *
 */
class ErrorFunction(EFType errFuncType, T, bool par=true): func if(isDataType!T)
{

  /*
   * Keep data immutable as it may be shared across threads.
   */
  alias immutable(T) iData;
  alias typeof(T.getPoint(0)) iDataPoint;

  private feedforwardnetwork net;
  private Regulizer reg;
  private immutable size_t numParms;
  private iData data;
  private double error = double.max;
  private double[] grad = null;

  /**
   * Params:
   * inNet = The network for which the error will be evaluated.
   * data  = The data set on which the error will be evaluated.
   * reg   = The Regulizer to apply to the network. This can be null if no
   *         regularization is desired.
   */
  public this(feedforwardnetwork inNet, iData data, Regulizer reg = null)
  {
    this.net = inNet.dup;
    this.reg = reg;
    this.data = data;

    this.numParms = net.parameters.length;
  }

  /**
   * Evaluate the error of the network with the given inputs as the weights and
   * and biases provided.
   *
   * Params:
   * inputs   = The values to set as weights in the network.
   * evalGrad = true if you will need to retrieve the gradient with respect to
   *            the provided weights.
   */
  public final override void evaluate(in double[] inputs, bool evalGrad = true)
  {

    // Copy in the parameters to the network
    net.parameters = inputs.dup;
    
    // Keep track of the count so you can average the error later
    size_t count = 0;

    // Initialize values
    error = 0.0;
    if(evalGrad && !grad)
    {
      grad = uninitializedArray!(double[])(numParms);
      grad[] = 0.0;
    }
    else if(evalGrad) grad[] = 0.0;
    else grad = null;

    /*==========================================================================
      Nested structure to hold the results of an error calculation over a range.
    ==========================================================================*/
    struct results {
      public size_t r_count;
      public double r_error;
      public double[] r_grad;

      public this(const size_t cnt, const double err, const double[] grd)
      {
        this.r_count = cnt;
        this.r_error = err;
        if(evalGrad) this.r_grad = grd.dup;
        else this.r_grad = null;
      }
    }

    /*==========================================================================
      Nested function to calculate the error over a range.
    ==========================================================================*/
    results doErrorChunk(DR)(DR dr, feedforwardnetwork nt)
    if(isDataRangeType!DR)
    {
      // Set up return variables
      size_t d_count = 0;
      double d_error = 0.0;
      double[] d_grad = null;
      if(evalGrad)
      {
        d_grad = uninitializedArray!(double[])(numParms);
        d_grad[] = 0.0;
      }

      foreach(dp; dr)
      {
        // Evaluate the network at the given points
        const(double[]) y = nt.eval(dp.inputs);

        // Calculate the error for the given point.
        static if(errFuncType == EFType.ChiSquare)
        {
          double err = 0.0;
          foreach(i; 0 .. y.length)
          { 
            double val = (y[i] - dp.targets[i]);
            err += val * val;
          }
          d_error += 0.5 * err;
        }

        else static if(errFuncType == EFType.CrossEntropy2C)
        {
          foreach(i; 0 .. y.length)
            d_error -= dp.targets[i] * log(y[i]) + 
                                        (1.0 - dp.targets[i]) * log(1.0 - y[i]);
        }

        else static if(errFuncType == EFType.CrossEntropy)
        {
          foreach(i; 0 .. y.length)
            d_error -= dp.targets[i] * log(y[i]);
        }

        else
        {
          pragma(msg, "Invalid EFType.");
          static assert(0);
        }
        
        // Do the back-propagation, assuming output activation function and
        // error functions are properly matched.
        if(evalGrad) d_grad[] += nt.backProp(dp.targets)[];

        // Increment the count
        ++d_count;
      }

      return results(d_count, d_error, d_grad);
    }

    static if(par) // Parallel Code
    {
      import std.parallelism;

      // How many threads to use?
      size_t numThreads = totalCPUs - 1;
      if(numThreads < 1) numThreads = 1;

      alias typeof(data.simpleRange) RngType;
      results[] reses = new results[numThreads];

      foreach(i, ref res; parallel(reses))
      {
        res = doErrorChunk!RngType(data.batchRange(i + 1, numThreads), net.dup);
      }

      foreach(i, res; reses)
      {
        count += res.r_count;
        error += res.r_error;
        if(evalGrad) grad[] += res.r_grad[];
      }
    }
    
    else // Serial Code
    {
      // Get a data range for iterating the data points
      auto dr = data.simpleRange;

      results res = doErrorChunk!(typeof(dr))(dr, net);

      count = res.r_count;
      error = res.r_error;
      if(evalGrad) grad[] = res.r_grad[];
    }

    // Average the error and the gradient.
    error /= count;
    if(evalGrad) grad[] /= count;


    // Add in the regularization error
    if(reg)
    {
      reg.evaluate(net.nonBiasParameters, evalGrad);
      error += reg.value;
      if(evalGrad) grad[] += reg.gradient[];
    }
  }
  
  /**
   * Returns: The value of the error as calculated by the last call to 
   *          evaluate.
   */
  public final override @property double value(){ return error; }
  
  /**
   * Returns: The value of the gradient as calculated by the last call to
   *          evaluate.
   */
  public final override @property double[] gradient(){ return grad.dup; }
}

version(unittest)
{
  import dffann.linearNetworks;

  // Test Data
  double[][] fakeData = 
                        [[  1.0,   2.0,   3.0,   4.0,  35.0,  31.0],
                         [  1.0,   3.0,   5.0,   7.0,  55.0,  47.0],
                         [  1.0,   1.0,   1.0,   1.0,  15.0,  15.0],
                         [ -1.0,   4.0,   2.0,  -2.0,  10.0,  14.0]];

  // All binary flags are false, because none of the data is binary!
  bool[] binaryFlags = [false, false, false, false, false, false];

  // Number of inputs and outputs
  enum numIn = 4;
  enum numOut = 2;
  
  // Normalize the data set (NO!, the predetermined weights for this data set 
  // don't allow it.)
  enum normalize = false; 

  // short hand for dealing with data
  alias immutable(Data!(numIn, numOut)) iData;

}

unittest
{
  mixin(dffann.dffann.announceTest("ChiSquareEF"));

  // Make a data set
  iData d1 = new iData(fakeData, binaryFlags, normalize);

  // Now, build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  alias ErrorFunction!(EFType.ChiSquare, iData, false) ChiSquareEF_S;
  alias ErrorFunction!(EFType.ChiSquare, iData) ChiSquareEF_P;

  ChiSquareEF_S ef_S = new ChiSquareEF_S(slprn, d1);
  ChiSquareEF_P ef_P = new ChiSquareEF_P(slprn, d1);

  // Test without regularization - evaluate the gradient
  ef_S.evaluate(slprn.parameters);
  ef_P.evaluate(slprn.parameters);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));
  assert(ef_S.grad !is null);
  assert(ef_P.grad !is null);

  // Test without regularization - do not evaluate the gradient
  ef_S.evaluate(slprn.parameters, false);
  ef_P.evaluate(slprn.parameters, false);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(ef_S.grad is null, format("%s",ef_S.grad));
  assert(ef_P.grad is null, format("%s",ef_P.grad));
}

/**
 * An abstract class for Regulizers. It includes methods for manipulating the
 * hyper-parameters so the training process itself can be optimized.
 */
abstract class Regulizer: func
{
  protected double errorTerm = double.max;
  protected double[] gradientTerm = null;

  /**
   * Returns: The hyper-parameters packed into an array.
   */
  public abstract @property double[] hyperParameters();

  /**
   * Set the value of the hyper-parameters.
   */
  public abstract @property void hyperParameters(in double[] hParms);

  /**
   * Returns: The value of the error as calculated by the last call to evaluate,
   *          which is required by the func interface.
   */
  public final override @property double value()
  {
    return errorTerm;
  }

  /**
   * Returns: The value of the error gradient as calculated by the last call to
   *          evaluate, which is required by the func interface.
   */
  public final override @property double[] gradient()
  {
    return gradientTerm;
  }

  /**
   * Required method by func interface, will be implemented in sub-classes.
   */
  public abstract void evaluate(in double[] inputs, bool grad=true);
}

/**
 * Penalizes large weights by adding a term proportional to the sum-of-squares 
 * of the weights.
 */
class WeightDecayRegulizer: Regulizer
{
  private double nu;

  /**
   * Params:
   * nuParm = the proportionality value. Should be between 0 and 1 in most
   * use cases, since all the errors are averaged over the number of points and
   * and the regularizations are are averaged over the number of weights.
   */
  public this(double nuParm)
  {
    this.nu = nuParm;
  }

  public final override void evaluate(in double[] inputs, bool grad=true)
  {
    // When optimizing the hyper parameters, they may go negative, so always
    // use the absolute value to force it to be positive.
    double pnu = fabs(nu);

    // Calculate the error term
    errorTerm = 0.0;
    foreach(i; 0 .. inputs.length)
    {
      errorTerm += inputs[i] * inputs[i];
    }
    errorTerm *= pnu / 2.0 / inputs.length;

    // Calculate the gradient...
    if(grad)
    {
      gradientTerm = uninitializedArray!(double[])(inputs.length);
      gradientTerm[] = pnu * inputs[] / inputs.length;
    }
  }

  public override @property double[] hyperParameters()
  {
    double[] toRet = new double[1];
    toRet[0] = nu;
    return toRet;
  }

  public override @property void hyperParameters(in double[] hParms)
  {
    assert(hParms.length == 1,"Invalid number of parameters.");
    this.nu = hParms[0];
  }
}
unittest
{
  mixin(dffann.dffann.announceTest("WeightDecayRegularizer"));

  // Build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  // Create a regulizer and evaluate it.
  WeightDecayRegulizer wdr = new WeightDecayRegulizer(0.1);
  wdr.evaluate(slprn.parameters, true);

  assert(approxEqual(wdr.value, 0.55));
  assert(approxEqual(wdr.gradient,
   [0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]));

  // Test the hyper-parameters
  assert(wdr.hyperParameters == [0.1]);
}

/**
 * Similar to weight decay Regularization, except when weights are much less
 * than nuRef they are driven to zero, and are thus effectively eliminated.
 */
class WeightEliminationRegulizer: Regulizer
{
  private double nu;
  private double nuRef;

  /** 
   * Params:
   * nuParm    = Proportionality parameter, same as in weight decay regulizer.
   * nuRefParm = A reference value to set the 'ideal' scale for the weights. The
   *             value of this parameter should be of order unity, loosely.
   *             If it is too much more or less it can cause instability in the
   *             training process.
   */
  public this(double nuParm, double nuRefParm)
  {
    this.nu = nuParm;
    this.nuRef = nuRefParm;
  }

  public final override void evaluate(in double[] inputs, bool grad=true)
  {
    // When optimizing the hyper parameters, they may go negative, so always
    // use the absolute value to force it to be positive.
    double pnu = fabs(nu);
    
    // Initialize
    errorTerm = 0.0;
    if(grad) gradientTerm = uninitializedArray!(double[])(inputs.length);

    // Calculate
    foreach(i; 0 .. inputs.length)
    {
      double w2 = inputs[i] * inputs[i];
      double denom = w2 + nuRef * nuRef;
      errorTerm += w2 / denom;
      if(grad)
      {
        gradientTerm[i] = pnu * inputs[i] * nuRef * nuRef / ( denom * denom) / inputs.length;
      }
    }
    errorTerm *= pnu / 2.0 / inputs.length;

  }

  public override @property double[] hyperParameters()
  {
    double[] toRet = new double[2];
    toRet[0] = nu;
    toRet[1] = nuRef;
    return toRet;
  }

  public override @property void hyperParameters(in double[] hParms)
  {
    assert(hParms.length == 2,"Invalid number of parameters.");
    this.nu = hParms[0];
    this.nuRef = hParms[1];
  }
}
unittest
{
  mixin(dffann.dffann.announceTest("WeightEliminationRegularizer"));

  // Build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  // Create a regulizer and evaluate it.
  WeightEliminationRegulizer wer = new WeightEliminationRegulizer(0.1, 1.0);
  wer.evaluate(slprn.parameters, true);

  assert(approxEqual(wer.value, 0.0410271), format("%s",wer.value));
  assert(approxEqual(wer.gradient,
                     [0.0025, 0.0008, 0.0003, 0.000138408, 7.39645e-05, 
                      7.39645e-05, 0.000138408, 0.0003, 0.0008, 0.0025]),
         format("%s",wer.gradient));

  // Test the hyper-parameters
  assert(wer.hyperParameters == [0.1, 1.0]);
}

unittest
{
  mixin(dffann.dffann.announceTest("ErrorFunction with Regulizer"));

  // Make a data set
  iData d1 = new iData(fakeData, binaryFlags, normalize);

  // Now, build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  alias ErrorFunction!(EFType.ChiSquare, iData, false) ChiSquareEF_S;
  alias ErrorFunction!(EFType.ChiSquare, iData) ChiSquareEF_P;

  auto wdr = new WeightDecayRegulizer(0.01);
  ChiSquareEF_S ef_S = new ChiSquareEF_S(slprn, d1, wdr);
  ChiSquareEF_P ef_P = new ChiSquareEF_P(slprn, d1, wdr);

  ef_S.evaluate(slprn.parameters);
  ef_P.evaluate(slprn.parameters);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));
  assert(ef_S.grad !is null);
  assert(ef_P.grad !is null);

  ef_S.evaluate(slprn.parameters, false);
  ef_P.evaluate(slprn.parameters, false);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(ef_S.grad is null, format("%s",ef_S.grad));
  assert(ef_P.grad is null, format("%s",ef_P.grad));
}
