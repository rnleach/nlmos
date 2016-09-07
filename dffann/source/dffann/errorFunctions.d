/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: February 7, 2015
 *
 * This module contains error functions AND regulizers. Regulizers are treated
 * as an addition to the error function.
 *
 */
module dffann.errorfunctions;

import std.array;
import std.math;
import std.range;

import numeric.numeric;
import numeric.func;

import dffann.data;
import dffann.feedforwardnetwork;
import dffann.strategy;

version(unitttest)
{
  import std.stdio;
  import std.string;
}

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
enum EFType {ChiSquare, CrossEntropy, CrossEntropy2C}

/**
 * A class for computing the value and gradient of an error function for 
 * a feed forward network with a given set of weights. This class implements 
 * the numeric.func.Func interface so it can be used with the function 
 * minimization tools in the numeric package during training.
 *
 * Params:
 * eft       = The specific error function to calculate.
 * para      = ParallelStrategy.parallel if you want to use the parallel 
 *             (concurrent) code in the evaluate method. You may want to use 
 *             ParallelStrategy.serial if you plan to parallelize at a higher 
 *             level construct.
 * batchMode = BatchStrategy.minibatch if you want to use mini-batches, or a 
 *             subset of the data for each iteration. Otherwise the default is
 *             BatchStrategy.batch to use the whole data set.
 * randomize = RandomStrategy.inOrder if you want to go through all the points
 *             in a data set in order. This is ignored (defaults to inOrder)
 *             UNLESS mini-batches are used. Then the extra effort is worth it 
 *             to default use a random strategy to prevent odd cycles in the 
 *             data.
 *
 */
class ErrorFunction(EFType eft, 
                    ParallelStrategy para    = ParallelStrategy.parallel,
                    BatchStrategy batchMode  = BatchStrategy.batch, 
                    RandomStrategy random    = RandomStrategy.random): Func 
{
  alias iData = immutable(Data);

  enum par = para == ParallelStrategy.parallel;
  enum useBatches = batchMode == BatchStrategy.minibatch;
  enum randomize = random == RandomStrategy.random;

  private FeedForwardNetwork net;
  private Regulizer reg;
  private immutable size_t numParms;
  private iData data;
  private double error = double.max;
  private double[] grad = null;

  
  static if(useBatches)
  {
    /**
     * Set and use the batch size when useBatches = true
     */
    public uint batchSize = 10;

    /*
      Choose the range type if doing batches, a randomized order is better to
      avoid cycles developing during training. If using an infinite range it 
      needs to be persistent between calls to evaluate.
     */
    static if(randomize)
    {
      private typeof(data.randomRange) infRange;
    }
    else
    {
      private typeof(data.infiniteRange) infRange;
    }
  }


  /**
   * Params:
   * inNet = The network for which the error will be evaluated.
   * data  = The range set on which the error will be evaluated.
   * reg   = The Regulizer to apply to the network. This can be null if no
   *         regularization is desired.
   */
  public this(FeedForwardNetwork inNet, iData data, Regulizer reg = null)
  {
    this.net = inNet.dup;
    this.reg = reg;
    this.data = data;

    this.numParms = net.parameters.length;

    static if(useBatches) 
    {
      static if(randomize)
      {
        this.infRange = data.randomRange;
      }
      else
      {
        this.infRange = data.infiniteRange;
      }
    }
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
    results doErrorChunk(DR)(DR dr, FeedForwardNetwork nt)
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
        static if(eft == EFType.ChiSquare)
        {
          double err = 0.0;
          foreach(i; 0 .. y.length)
          { 
            const double val = (y[i] - dp.targets[i]);
            err += val * val;
          }
          d_error += 0.5 * err;
        }

        else static if(eft == EFType.CrossEntropy2C)
        {
          d_error -= dp.targets[0] * log(y[0]) + 
                                        (1.0 - dp.targets[0]) * log(1.0 - y[0]);
        }

        else static if(eft == EFType.CrossEntropy)
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
      import std.parallelism : totalCPUs, parallel;

      // How many threads to use?
      size_t numThreads = totalCPUs - 1;
      if(numThreads < 1) numThreads = 1;

      results[] reses = new results[numThreads];

      static if(!useBatches)
      {
        auto myRange = this.data.simpleRange;
      }
      else
      {
        auto myRange = take(infRange, batchSize);
      }

      auto drs = evenChunks(myRange, numThreads);

      alias Rng = typeof(drs.front);
      Rng[] dra = array(drs);

      foreach(i, ref res; parallel(reses))
      {
        res = doErrorChunk(dra[i], net.dup);
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
      static if(!useBatches)
      {
        auto myRange = this.data.simpleRange;
      }
      else
      {
        auto myRange = take(infRange, batchSize);
      }

      results res = doErrorChunk(myRange, net);

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
  import dffann.linearnetworks;

  // Test Data
  double[][] fakeData = 
                        [[  1.0,   2.0,   3.0,   4.0,  35.0,  31.0],
                         [  1.0,   3.0,   5.0,   7.0,  55.0,  47.0],
                         [  1.0,   1.0,   1.0,   1.0,  15.0,  15.0],
                         [ -1.0,   4.0,   2.0,  -2.0,  10.0,  14.0]];

  // Number of inputs and outputs
  enum numIn = 4;
  enum numOut = 2;

}

///
unittest
{
  // ChiSquareEF

  // Make a data set
  auto d1 = Data.createImmutableData(numIn, numOut, fakeData);

  // Now, build a network.
  const double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  alias ChiSquareEF_P  = ErrorFunction!(EFType.ChiSquare);

  alias ChiSquareEF_PC = ErrorFunction!(EFType.ChiSquare, 
                                        ParallelStrategy.parallel, 
                                        BatchStrategy.minibatch, 
                                        RandomStrategy.inOrder);
  
  alias ChiSquareEF_S  = ErrorFunction!(EFType.ChiSquare, 
                                        ParallelStrategy.serial);

  alias ChiSquareEF_SC = ErrorFunction!(EFType.ChiSquare, 
                                        ParallelStrategy.serial, 
                                        BatchStrategy.minibatch, 
                                        RandomStrategy.inOrder);

  ChiSquareEF_S ef_S  = new ChiSquareEF_S(slprn, d1);
  ChiSquareEF_P ef_P  = new ChiSquareEF_P(slprn, d1);
  ChiSquareEF_SC ef_SC = new ChiSquareEF_SC(slprn, d1);
  ChiSquareEF_PC ef_PC = new ChiSquareEF_PC(slprn, d1);
  ef_SC.batchSize = ef_PC.batchSize = 2;

  // Test without regularization - evaluate the gradient
  ef_S.evaluate(slprn.parameters);
  ef_P.evaluate(slprn.parameters);
  ef_SC.evaluate(slprn.parameters);
  ef_PC.evaluate(slprn.parameters);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(approxEqual(ef_S.grad, ef_P.grad));
  assert(approxEqual(ef_SC.value, ef_PC.value));
  assert(approxEqual(ef_SC.grad, ef_PC.grad));
  assert(ef_S.grad  !is null);
  assert(ef_P.grad  !is null);
  assert(ef_SC.grad !is null);
  assert(ef_PC.grad !is null);

  // Test without regularization - do not evaluate the gradient
  ef_S.evaluate(slprn.parameters, false);
  ef_P.evaluate(slprn.parameters, false);
  ef_SC.evaluate(slprn.parameters, false);
  ef_PC.evaluate(slprn.parameters, false);

  assert(approxEqual(ef_S.value, ef_P.value));
  assert(approxEqual(ef_SC.value, ef_PC.value));
  assert(ef_S.grad is null, format("%s",ef_S.grad));
  assert(ef_P.grad is null, format("%s",ef_P.grad));
  assert(ef_SC.grad is null, format("%s",ef_SC.grad));
  assert(ef_PC.grad is null, format("%s",ef_PC.grad));

  // Try again, this time with an error function that is not zero!
  const double[] wts2 = [-1.0, 2.0, -3.0, 4.0, -5.0, 5.0, -4.0, 3.0, -2.0, 1.0];
  slprn.parameters = wts2;

  alias ChiSquareEF_PCR = ErrorFunction!(EFType.ChiSquare, 
                                        ParallelStrategy.parallel, 
                                        BatchStrategy.minibatch, 
                                        RandomStrategy.random);

  alias ChiSquareEF_SCR = ErrorFunction!(EFType.ChiSquare, 
                                        ParallelStrategy.serial, 
                                        BatchStrategy.minibatch, 
                                        RandomStrategy.random);

  ChiSquareEF_SCR ef_SCR = new ChiSquareEF_SCR(slprn, d1);
  ChiSquareEF_PCR ef_PCR = new ChiSquareEF_PCR(slprn, d1);
  ef_SCR.batchSize = ef_PCR.batchSize = ef_SC.batchSize = ef_PC.batchSize;

  // Test without regularization - evaluate the gradient
  ef_S.evaluate(   slprn.parameters );
  ef_P.evaluate(   slprn.parameters );
  ef_SC.evaluate(  slprn.parameters );
  ef_PC.evaluate(  slprn.parameters );
  ef_SCR.evaluate( slprn.parameters );
  ef_PCR.evaluate( slprn.parameters );

  assert( approxEqual( ef_S.value,  ef_P.value  ));
  assert( approxEqual( ef_S.grad,   ef_P.grad   ));
  assert( approxEqual( ef_SC.value, ef_PC.value ));
  assert( approxEqual( ef_SC.grad,  ef_PC.grad  ));

  // Test without regularization - do not evaluate the gradient
  ef_S.evaluate(   slprn.parameters, false );
  ef_P.evaluate(   slprn.parameters, false );
  ef_SC.evaluate(  slprn.parameters, false );
  ef_PC.evaluate(  slprn.parameters, false );
  ef_SCR.evaluate( slprn.parameters, false );
  ef_PCR.evaluate( slprn.parameters, false );

  assert( approxEqual( ef_S.value,  ef_P.value  ));
  assert( approxEqual( ef_SC.value, ef_PC.value ));
}

/**
 * An abstract class for Regulizers. It includes methods for manipulating the
 * hyper-parameters so the training process itself can be optimized.
 */
abstract class Regulizer: Func
{
  protected double errorTerm = double.max;
  protected double[] gradientTerm = null;

  /**
   * Returns: The hyper-parameters packed into an array.
   */
  public abstract @property const(double[]) hyperParameters() const;

  /**
   * Set the value of the hyper-parameters.
   */
  public abstract @property void hyperParameters(in double[] hParms);

  /**
   * Returns: The value of the error as calculated by the last call to evaluate,
   *          which is required by the Func interface.
   */
  public final override @property double value() pure
  {
    return errorTerm;
  }

  /**
   * Returns: The value of the error gradient as calculated by the last call to
   *          evaluate, which is required by the Func interface.
   */
  public final override @property double[] gradient() pure
  {
    return gradientTerm;
  }

  /**
   * Required method by Func interface, will be implemented in sub-classes.
   */
  public abstract void evaluate(in double[] inputs, bool grad=true) pure;
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

  public final override void evaluate(in double[] inputs, bool grad=true) pure
  {
    // When optimizing the hyper parameters, they may go negative, so always
    // use the absolute value to force it to be positive.
    const double pnu = fabs(nu);

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

  public override @property const(double[]) hyperParameters() const
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

///
unittest
{
  // WeightDecayRegularizer

  // Build a network.
  const double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
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

  public final override void evaluate(in double[] inputs, bool grad=true) pure
  {
    // When optimizing the hyper parameters, they may go negative, so always
    // use the absolute value to force it to be positive.
    const double pnu = fabs(nu);
    
    // Initialize
    errorTerm = 0.0;
    if(grad) gradientTerm = uninitializedArray!(double[])(inputs.length);

    // Calculate
    foreach(i; 0 .. inputs.length)
    {
      const double w2 = inputs[i] * inputs[i];
      const double denom = w2 + nuRef * nuRef;
      errorTerm += w2 / denom;
      if(grad)
      {
        gradientTerm[i] = pnu * inputs[i] * nuRef * nuRef / ( denom * denom) / inputs.length;
      }
    }
    errorTerm *= pnu / 2.0 / inputs.length;

  }

  public override @property const(double[]) hyperParameters() const
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

///
unittest
{
  // WeightEliminationRegularizer

  // Build a network.
  const double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
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

///
unittest
{
  // ErrorFunction with Regulizer

  // Make a data set
  auto d1 = Data.createImmutableData(numIn, numOut, fakeData);

  // Now, build a network.
  const double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  alias ChiSquareEF_S = ErrorFunction!(EFType.ChiSquare, 
                                       ParallelStrategy.serial);

  alias ChiSquareEF_P = ErrorFunction!(EFType.ChiSquare);

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
