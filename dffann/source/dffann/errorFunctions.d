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
    if(evalGrad)
    {
      grad = uninitializedArray!(double[])(numParms);
      grad[] = 0.0;
    }

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
      double[] d_grad;
      if(evalGrad)
      {
        d_grad = uninitializedArray!(double[])(numParms);
        d_grad[] = 0.0;
      }

      foreach(dp; dr)
      {
        // Evaluate the network at the given points
        double[] y = nt.eval(dp.inputs);

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

  alias immutable(Data!(5,2)) iData;

  // Test Data
  double[][] testData = 
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
     [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
     [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
     [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
     [1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4],
     [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
     [1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6],
     [1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7],
     [1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
     [1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9]];

  bool[] flags = [false, false, false, false, false, false, false]; 
}
unittest
{
  // TODO - make known values to test
  mixin(dffann.dffann.announceTest("ChiSquareEF"));

  import dffann.linearNetworks;

  iData myData = new iData(testData, flags);

  LinRegNet slprn = new LinRegNet(5,2);

  alias ErrorFunction!(EFType.ChiSquare, iData, false) ChiSquareEF_S;
  alias ErrorFunction!(EFType.ChiSquare, iData) ChiSquareEF_P;

  ChiSquareEF_S ef_S = new ChiSquareEF_S(slprn, myData);
  ChiSquareEF_P ef_P = new ChiSquareEF_P(slprn, myData);

  // Test without regularization
  ef_S.evaluate(slprn.parameters);
  ef_P.evaluate(slprn.parameters);

  assert(approxEqual(ef_S.value, ef_P.value),
    format("\n\nSerial=%f != Parallel=%f\n\n", ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));

  ef_S.evaluate(slprn.parameters, false);
  ef_P.evaluate(slprn.parameters, false);

  assert(approxEqual(ef_S.value, ef_P.value),
    format("\n\nSerial=%f != Parallel=%f\n\n", ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));

  // Test with regularization
  auto wdr = new WeightDecayRegulizer(0.01);
  ef_S = new ChiSquareEF_S(slprn, myData, wdr);
  ef_P = new ChiSquareEF_P(slprn, myData, wdr);

  ef_S.evaluate(slprn.parameters);
  ef_P.evaluate(slprn.parameters);

  assert(approxEqual(ef_S.value, ef_P.value),
    format("\n\nSerial=%f != Parallel=%f\n\n", ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));

  ef_S.evaluate(slprn.parameters, false);
  ef_P.evaluate(slprn.parameters, false);

  assert(approxEqual(ef_S.value, ef_P.value),
    format("\n\nSerial=%f != Parallel=%f\n\n", ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));
}

/**
 * An abstract class for Regulizers. It includes methods for manipulating the
 * hyper-parameters to the training process itself can be optimized.
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
   * Calculate the gradient of the error with respect to the Regularizations
   * hyper-parameters.
   * 
   * Params:
   * inputs = The network weights, the same values you would pass to the error
   *          function for evaluation. Note that often in regularizations you
   *          want to ignore the bias parameters, so ensure they are set to 
   *          zero by using the nonBiasParms method of the network to fetch
   *          them.
   *
   * Returns: The gradient of the error function with respect to the hyper-
   *          parameters.
   */
  public abstract double[] hyperGradient(in double[] inputs);

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
 * Penalizes large weights by add a term proportional to the sum-of-squares of 
 * the weights.
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

  public override double[] hyperGradient(in double[] inputs)
  {
    double[] hyperGrad = uninitializedArray!(double[])(1);
    hyperGrad[0] = 0.0;

    foreach(i; 0 .. inputs.length)
    {
      hyperGrad[0] += inputs[i] * inputs[i];
    }

    hyperGrad[0] /= 2.0 * sgn(nu) * inputs.length;

    return hyperGrad;
  }
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

   public override double[] hyperGradient(in double[] inputs)
  {
    double[] hyperGrad = uninitializedArray!(double[])(2);
    hyperGrad[] = 0.0;

    foreach(i; 0 .. inputs.length)
    {
      double w2 = inputs[i] * inputs[i];
      double denom = w2 + nuRef * nuRef;
      hyperGrad[0] += w2 / denom;
      hyperGrad[1] += w2 / (denom * denom);
    }

    hyperGrad[0] *= sgn(nu) / 2.0 / inputs.length;
    hyperGrad[1] *= -fabs(nu) * nuRef / inputs.length;

    return hyperGrad;
  }
}

