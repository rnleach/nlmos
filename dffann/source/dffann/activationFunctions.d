/**
* Activation function definitions.
*
* Activation functions should be structs with only static members and no 
* internal state. The required methods are:
*
* Examples:
* ---------
* static void eval(in double[] x, ref double[] y);
* static double[] deriv(in double[] x, in double[] y);
* ---------
*
* The eval function returns the activation function applied to the values in x
* in the array y. And deriv returns the gradient of each node y[] with respect 
* to its activation x[].
*
*/
module dffann.activationfunctions;

import numeric.numeric;

import std.array;
import std.math;
import std.traits;

/// Shorthand versions of some template tests.
alias isAF = isActivationFunction;
alias isOAF = isOutputActivationFunction; /// ditto

/**
* Assure a struct meets the requirements to be used as an activation function.
*/
template isActivationFunction(A)
{
  // Check that you can actually use it!
  enum bool canUse = is(typeof(
  (inout int = 0)
  {
      // Can evaluate and fill an array with a value
      double[] x = [0.0, 1.0, 2.0];
      double[] y = new double[3];

      // By calling with the type name, we are forcing these to be static
      // methods of a class or struct.
      A.eval(x, y);

      // Can evaluate a derivative
      double[] d = A.deriv(x,y);
      
  }));

  // If it can't be called, return short fused so we don't get compiler errors
  // on the sections below, which would be misleading.
  static if(!canUse)
  {
    enum bool isActivationFunction = canUse;
  }
  else
  {
    // Now check the parameters
    alias STC = ParameterStorageClass;
    alias pstcEval = ParameterStorageClassTuple!(A.eval);
    alias pstcDeriv = ParameterStorageClassTuple!(A.deriv);

    // Check which arguments should be input arguments
    enum bool argsIn = (pstcEval[0] == STC.none && 
                        pstcDeriv[0] == STC.none && 
                        pstcDeriv[1] == STC.none);

    // Check that y is a ref argument
    enum bool argsRef = (pstcEval[1] == STC.ref_);

    enum bool isActivationFunction = argsIn && argsRef && canUse;
  }
}

/**
* Assure an activation function is allowed to be on the output layer of a
* network.
*/
template isOutputActivationFunction(A)
{

  // Check that it is an activation function first!
  enum bool validAF = isAF!A;

  static if(!validAF)
  {
    enum bool isOutputActivationFunction = false;
  }
  else
  {
    // Check if it is one of the allowed types.
    enum bool allowedOutputAF = is(A == LinearAF) ||
                                is(A == SigmoidAF) ||
                                is(A == SoftmaxAF);

    enum bool isOutputActivationFunction = allowedOutputAF;
  }
}

/**
* Linear activation function, just returns its activation with no-nonlinear
* transformation applied.
*/
public struct LinearAF
{
  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] act, ref double[] outpt)
  {
    outpt[] = act[];
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] act, in double[] outpt)
  {
    assert(act.length == outpt.length);

    double[] toRet = uninitializedArray!(double[])(act.length);
    toRet[] = 1.0;
    return toRet;
  }
}

unittest
{
  mixin(announceTest("LinearAF"));

  const double[] input = [1.0, 2.0, 3.0, 42.0];
  double[] output = [0.0, 0.0, 0.0, 0.0];
  LinearAF.eval(input, output);

  // Test eval
  assert(approxEqual(input, output));

  // Test deriv
  assert(approxEqual(LinearAF.deriv(input, output), [1.0, 1.0, 1.0, 1.0]));
}

static assert(isAF!LinearAF);
static assert(isOAF!LinearAF);

/**
* Sigmoid activation function.
*
* For each x[i] calculates y[i] = 1.0 / (1.0 + exp(-x[i])). The derivative for
* this activation function only depends on the y vector, and is calculated as
* deriv[i] = y[i] * (1.0 - y[i]).
*/
public struct SigmoidAF
{
  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] act, ref double[] outpt)
  {
    foreach(i; 0 .. outpt.length)
      outpt[i] = 1.0 / (1.0 + exp(-act[i]));
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] act, in double[] outpt)
  {
    double[] derivative = uninitializedArray!(double[])(act.length);

    derivative[] = outpt[] * (1.0 - outpt[]);

    return derivative;
  }
}

unittest
{
  mixin(announceTest("SigmoidAF"));
  
  const double[] input = [0.0, 1000.0, -1000.0];
  double[] output = [0.0, 0.0, 0.0];
  SigmoidAF.eval(input, output);

  // Test eval
  assert(approxEqual(output, [0.5, 1.0, 0.0]));

  // Test deriv
  assert(approxEqual(SigmoidAF.deriv(input, output), [0.25, 0.0, 0.0]));
}

static assert(isAF!SigmoidAF);
static assert(isOAF!SigmoidAF);

/**
* Tanh activation function.
*/
public struct TanhAF
{
  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] act, ref double[] outpt)
  {
    foreach(i; 0 .. outpt.length)
      outpt[i] = tanh(act[i]);
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] act, in double[] outpt)
  {
    double[] derivative = uninitializedArray!(double[])(act.length);

    derivative[] = 1.0 - outpt[] * outpt[];

    return derivative;
  }
}
static assert(isAF!TanhAF);

/**
* arctan activation function.
*/
public struct ArctanAF{

  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] act, ref double[] outpt)
  {
    foreach(i; 0 .. outpt.length)
      outpt[i] = atan(act[i]);
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] act, in double[] outpt)
  {
    double[] derivative = uninitializedArray!(double[])(act.length);

    derivative[] = 1.0 / (1.0 + outpt[] * outpt[]);

    return derivative;
  }
}
static assert(isAF!ArctanAF);

/**
* This is the softplus activation function. It should help in the training 
* of deeper network architectures because the derivative does not saturate
* as quickly.
*/
public struct SoftPlusAF
{

  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] act, ref double[] outpt)
  {
    foreach(i; 0 .. outpt.length)
      outpt[i] = act[i] / (1.0 + abs(act[i]));
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] act, in double[] outpt)
  {
    double[] derivative = uninitializedArray!(double[])(act.length);

    foreach(i; 0 .. outpt.length)
    {
      const double absV = abs(act[i]);
      const double k = 1.0 / (1.0 + absV);
      derivative[i] = k * (1.0 - k * absV);
    }

    return derivative;
  }
}
static assert(isAF!SoftPlusAF);

/**
* The softmax vector for vector x.
*
*   For multiple classification, if you use this as the output activation 
*   function then all the outputs will range between 0 and 1 and will sum to 1.
*   They can be considered the posterior probability of an input being in each
*   class.
*
*   Note: If x is a single element vector, this will always be 1. So this
*   should not be used for the output activation of a network with 1 output,
*   otherwise it will always be 1.0!
*/
public struct SoftmaxAF
{
  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] x, ref double[] y)
  {
    
    // Softmax is unaltered under a shift to all elements of the list, so
    //  shift left to avoid overflow.
    double shift = 0.0;
    foreach(val; x)
      if(val > shift) shift = val;
    
    double denom = 0.0;
    foreach(val; x)
      denom += exp(val - shift);
    
    // If nothing has a high probability...what to do?
    // Add a constant to each denominator to prevent overflow/underflow
    if(denom < 1.0e-6)
    { 
      denom = 0.0;
        foreach(val; x)
          denom += exp(val - shift) + 1.0;
        foreach(i; 0 .. x.length)
        {
          y[i] = (exp(x[i] - shift) + 1.0)/denom;
        }
    }
    
    else
        foreach(i; 0 .. x.length)
          y[i] = exp(x[i] - shift)/denom;
  }

  /**
  * See module documentation for overview of activation functions.
  */
  public static double[] deriv(in double[] x, in double[] y)
  {
    double[] s = uninitializedArray!(double[])(y.length);

    s[] = y[] * (1.0 - y[]);

    return s;
  }
}
static assert(isAF!SoftmaxAF);
static assert(isOAF!SoftmaxAF);

/**
* This activation function returns it's outputs if they are greater than zero,
* or just zero otherwise.
*
* Some research(CITE) has shown that deep networks with this kind of activation
* function can be useful due to the sparsity that naturally arises. However,
* it takes a lot of neurons, a lot more than you would typically need with
* other activation function types.
*/
public struct RectifiedLinearAF
{

  /**
  * See module documentation for overview of activation functions.
  */
  public static void eval(in double[] x, ref double[] y)
  {

    foreach(i; 0 .. x.length)
      y[i] = x[i] < 0.0 ? 0.0 : x[i];
  }

  /**
  * See module documentation for overview of activation functions.
  */ 
  public static double[] deriv( in double[] x, in double[] y)
  {
    double[] s = uninitializedArray!(double[])(y.length);

    foreach(i; 0 .. x.length)
      s[i] = y[i] > 0.0 ? 1.0 : 0.0;
    
    return s;
  }
}
static assert(isAF!RectifiedLinearAF);
