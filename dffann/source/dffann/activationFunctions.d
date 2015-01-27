/**
 * Activation functions definitions.
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
 * in the array y. And deriv returns the gradient with respect to each node
 * y[] with respect to its activation x[].
 *
 */
module dffann.activationFunctions;

public import dffann.dffann;

import std.math;
import std.traits;


alias isActivationFunction isAF;
alias isOutputActivationFunction isOAF;
/**
 * Assure a struct meets the requirements to be used as an activation function.
 */
template isActivationFunction(A){

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
  // on the sections beow, which would be mis-leading.
  static if(!canUse) enum bool isActivationFunction = canUse;

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

/**
 * Assure an activation function is allowed to be on the output layer of a
 * network.
 */
template isOutputActivationFunction(A){

  // Check that it is an activation function first!
  enum bool validAF = isAF!A;

  static if(!validAF) enum bool isOutputActivationFunction = false;

  // Check if it is one of the allowed types.
  enum bool allowedOutputAF = is(A == linearAF) ||
                              is(A == sigmoidAF) ||
                              is(A == softmaxAF);

  enum bool isOutputActivationFunction = allowedOutputAF;
}

/**
 * Linear activation function, just returns its activation with no-nonlinear
 * transformation applied.
 */
public struct linearAF{
  public static void eval(in double[] act, ref double[] outpt){
    outpt[] = act[];
  }

  public static double[] deriv(in double[] act, in double[] outpt){
    assert(act.length == outpt.length);

    double[] toRet = new double[act.length];
    toRet[] = 1.0;
    return toRet;
  }
}
static assert(isAF!linearAF);
static assert(isOAF!linearAF);

/**
 * Sigmoid activation function.
 *
 * For each x[i] calculates y[i] = 1.0 / (1.0 + exp(-x[i])). The derivative for
 * this activation function only depends on the y vector, and is calculated as
 * deriv[i] = y[i] * (1.0 - y[i]).
 */
public struct sigmoidAF{
  public static void eval(in double[] act, ref double[] outpt){
    foreach(i; 0 .. outpt.length)
      outpt[i] = 1.0 / (1.0 + exp(-act[i]));
  }

  public static double[] deriv(in double[] act, in double[] outpt){
    double[] derivative = new double[act.length];

    derivative[] = outpt[] * (1.0 - outpt[]);

    return derivative;
  }
}
static assert(isAF!sigmoidAF);
static assert(isOAF!sigmoidAF);

/**
 * Tanh activation function.
 */
public struct tanhAF{
  public static void eval(in double[] act, ref double[] outpt){
    foreach(i; 0 .. outpt.length)
      outpt[i] = tanh(act[i]);
  }

  public static double[] deriv(in double[] act, in double[] outpt){
    double[] derivative = new double[act.length];

    derivative[] = 1.0 - outpt[] * outpt[];

    return derivative;
  }
}
static assert(isAF!tanhAF);

/**
 * arctan activation function.
 */
public struct arctanAF{
  public static void eval(in double[] act, ref double[] outpt){
    foreach(i; 0 .. outpt.length)
      outpt[i] = atan(act[i]);
  }

  public static double[] deriv(in double[] act, in double[] outpt){
    double[] derivative = new double[act.length];

    derivative[] = 1.0 / (1.0 + outpt[] * outpt[]);

    return derivative;
  }
}
static assert(isAF!arctanAF);

/**
 * This is the softplus activation function. It should help in the training 
 * of deeper network architectures because the derivative does not saturate
 * as quickly.
 *
 */
public struct softPlusAF{
  public static void eval(in double[] act, ref double[] outpt){
    foreach(i; 0 .. outpt.length)
      outpt[i] = act[i] / (1.0 + abs(act[i]));
  }

  public static double[] deriv(in double[] act, in double[] outpt){
    double[] derivative = new double[act.length];

    foreach(i; 0 .. outpt.length){
      double absV = abs(act[i]);
      double k = 1.0 / (1.0 + absV);
      derivative[i] = k * (1.0 - k * absV);
    }

    return derivative;
  }
}
static assert(isAF!softPlusAF);

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
 *
 */
public struct softmaxAF{

  public static void eval(in double[] x, ref double[] y) {
    
    // Softmax is unaltered under a shift to all elements of the list, so
    //  shift left to avoid overflow.
    double shift = 0.0;
    foreach(val; x)
      if(val > shift) shift = val;
    
    double denom = 0.0;
    foreach(val; x)
      denom += exp(val - shift);
    
    // If nothing has a high probability...what to do?
    // Add a constant to each denom to prevent overflow/underflow
    if(denom < 1.0e-6){ 
      denom = 0.0;
        foreach(val; x)
          denom += exp(val - shift) + 1.0;
        foreach(i; 0 .. x.length){
          y[i] = (exp(x[i] - shift) + 1.0)/denom;
        }
    }
    
    else
        foreach(i; 0 .. x.length)
          y[i] = exp(x[i] - shift)/denom;
    
  }

  public static double[] deriv(in double[] x, in double[] y) {
    double[] s = new double[y.length];

    s[] = y[] * (1.0 - y[]);

    return s;
  }
}
static assert(isAF!softmaxAF);
static assert(isOAF!softmaxAF);

/**
 * This activation function returns it's outputs if they are greater than zero,
 * or just zero otherwise.
 *
 * Some research(CITE) has shown that deep networks with this kind of activation
 * function can be useful due to the sparsity that naturally arises. However,
 * it takes a lot of neurons, a lot more than you would typically need with
 * other activation function types.
 */
public struct rectifiedLinearAF{
  public static void eval(in double[] x, ref double[] y) {

    foreach(i; 0 .. x.length)
      y[i] = x[i] < 0.0 ? 0.0 : x[i];
  }

  public static double[] deriv( in double[] x, in double[] y) {
    double[] s = new double[y.length];

    foreach(i; 0 .. x.length)
      s[i] = y[i] > 0.0 ? 1.0 : 0.0;
    
    return s;
  }
}
static assert(isAF!rectifiedLinearAF);
