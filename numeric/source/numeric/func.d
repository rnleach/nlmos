/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: January 23, 2015
 */
module numeric.func;

public import numeric.numeric;

version(unittest)
{
  import std.math;
}

/**
 * Basic interface for a function that will be manipulated by other numeric
 * routines.
 */
interface Func
{

  /**
   * Evaluate the function with the given inputs, saving the
   * value and gradient for later retrieval.
   * 
   * Params:
   * inputs = postition to evaluate at, e.g. inputs = [x,y] for f(x,y).
   */
  public void evaluate(in double[] inputs, bool grad = false);
  
  /**
   * Returns: the value that resulted from the last call to 
   * evaluate.
   */
  public @property double value();
  
  /**
   * Returns: the gradient with respect to the provided inputs 
   * from the last call to evaluate. If the gradient was not
   * evaluated then it should return null.
   */
  public @property double[] gradient();
  
}

version(unittest)
{
  // Some functions to test on.
  import std.algorithm: reduce;

  class SquareMachine: Func{
    double result;
    double[] lgradient = null;

    override final void evaluate(in double[] inputs, bool grad = false)
    {
      this.result = reduce!"a + b * b"(0.0,inputs);

      if(grad){
        lgradient = new double[](inputs.length);
        lgradient[] = 2.0 * inputs[];
      }
    }

    override final @property double value(){return this.result;}
    override final @property double[] gradient(){return this.lgradient;}
  }

  /*
   * Function of 3 independent variables, evaluates to 
   * f = (x - 1)^2 + (y - 2)^2 + (z - 3)^2
   */
  class AnotherFunction: Func
  {
    double result;
    double[3] lgradient = double.nan;

    override final void evaluate(in double[] inputs, bool grad = false){
      assert(inputs.length == 3);

      this.result = pow(inputs[0] - 1.0,2) + 
                    pow(inputs[1] - 2.0,2) + 
                    pow(inputs[2] - 3.0,2);

      if(grad)
      {
        lgradient = new double[](3);
        lgradient[0] = 2.0* (inputs[0] - 1.0);
        lgradient[1] = 2.0* (inputs[1] - 2.0);
        lgradient[2] = 2.0* (inputs[2] - 3.0);
      }
    }

    override final @property double value(){return this.result;}
    override final @property double[] gradient(){return this.lgradient;}
  }

  /*
   * Function of 3 independent variables, evaluates to 
   * f = -((x - 1)^2 + (y - 2)^2 + (z - 3)^2)
   *
   * This function has no local minima, and the global minima is at infinity,
   * so it is nice for testing the failure of minimization routines!
   */
  class ANegativeFunction: Func
  {
    double result;
    double[3] lgradient = double.nan;

    override final void evaluate(in double[] inputs, bool grad = false){
      assert(inputs.length == 3);

      this.result = -(pow(inputs[0] - 1.0,2) + 
                      pow(inputs[1] - 2.0,2) + 
                      pow(inputs[2] - 3.0,2));

      if(grad)
      {
        lgradient = new double[](3);
        lgradient[0] = -2.0* (inputs[0] - 1.0);
        lgradient[1] = -2.0* (inputs[1] - 2.0);
        lgradient[2] = -2.0* (inputs[2] - 3.0);
      }
    }

    override final @property double value(){return this.result;}
    override final @property double[] gradient(){return this.lgradient;}
  }
}

unittest
{
  // Func

  SquareMachine sm = new SquareMachine;
  sm.evaluate([0.0,1.0,2.0],true);
  assert(approxEqual(sm.value,5.0));
  assert(approxEqual(sm.gradient,[0.0,2.0,4.0]));

  AnotherFunction af = new AnotherFunction;
  af.evaluate([0.0, 1.0, 2.0], true);
  assert(approxEqual(af.value, 3.0));
  assert(approxEqual(af.gradient, [-2.0, -2.0, -2.0]));
}