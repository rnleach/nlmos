/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: February 7, 2015
 */
module dffann.errorFunctions;

import std.math;

import numeric.func;

public import dffann.dffann;
import dffann.data;
import dffann.feedforwardnetwork;

/**
 * Regulizers are just functions, but create the alias for clarity of purpose in
 * code below.
 */
alias func Regulizer;

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
 * TODO
 */
class ErrorFunction(EFType errFuncType, T, bool par=true): func if(isDataType!T)
{

  /**
   * Keep data immutable as it may be shared across threads.
   */
  alias immutable(T) iData;
  alias typeof(T.getPoint(0)) iDataPoint;

  private feedforwardnetwork net;
  private Regulizer reg;
  private size_t numParms = 0;
  private iData data;
  private double error = double.max;
  private double[] grad;

  public this(feedforwardnetwork inNet, iData data, Regulizer reg = null)
  {
    this.net = inNet.dup;
    this.reg = reg;
    this.data = data;

    this.numParms = net.parameters.length;
    this.grad = new double[numParms];
  }

  public override void evaluate(double[] inputs, bool evalGrad = true)
  {
   
    static if(par)
    {
       // TODO - make parallel version
    }
    
    else
    {
      // Copy in the parameters to the network
      net.parameters = inputs.dup;
      
      // Keep track of the count so you can average the error later
      size_t count = 0;

      // Initialize values
      error = 0.0;
      grad[] = 0.0;

      // Get a data range for iterating the data points
      auto dr = data.simpleRange;

      foreach(dp; dr)
      {
        // Evaluate the network at the given points
        double[] y = net.eval(dp.inputs);

        // Calculate the error for the given point.
        static if(errFuncType == EFType.ChiSquare)
        {
          double err = 0.0;
          foreach(i; 0 .. y.length)
          { 
            double val = (y[i] - dp.targets[i]);
            err += val * val;
          }
          error += 0.5 * err;
        }

        else static if(errFuncType == EFType.CrossEntropy2C)
        {
          foreach(i; 0 .. y.length)
            error -= dp.targets[i] * log(y[i]) + 
                                          (1.0 - dp.targets[i]) * log(1.0 - y[i]);
        }

        else static if(errFuncType == EFType.CrossEntropy)
        {
          foreach(i; 0 .. y.length)
            error -= dp.targets[i] * log(y[i]);
        }

        else
        {
          pragma(msg, "Invalid EFType.");
          static assert(0);
        }
        
        // Do the back-propagation, assuming output activation function and
        // error functions are properly matched.
        grad[] += net.backProp(dp.targets)[];

        // Increment the count
        ++count;
      }
    }

    // Average the error and the gradient.
    error /= count;
    grad[] /= count;

    // Add in the regularization error
    if(reg)
    {
      reg.evaluate(net.nonBiasParameters, true);
      error += reg.value;
      grad[] = grad[] + reg.gradient[];
    }
  }
  
  public override @property double value(){ return error; }
  
  public override @property double[] gradient(){ return grad.dup; }
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

  alias ErrorFunction!(EFType.ChiSquare, iData, false) ChiSquareEF;

  ChiSquareEF ef = new ChiSquareEF(slprn, myData);

  ef.evaluate(slprn.parameters);
}