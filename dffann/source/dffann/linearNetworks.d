/**
 * Collection of linear networks. 
 *
 * Author: Ryan Leach
 */
module dffann.linearNetworks;

public import dffann.dffann;

import dffann.activationFunctions;
import dffann.feedforwardnetwork;

import numeric.random: gasdev;

import std.conv;
import std.datetime;
import std.exception;
import std.math;
import std.regex;
import std.string;

version(unittest) import dffann.data;

public class LinearNetwork(OAF) : feedforwardnetwork if(isOAF!OAF)
{

  private double[] weights;
  private double[] biases;
  private const(double)[] inputNodes;
  private double[] outputActivationNodes;
  static if(!is(OAF == linearAF))  private double[] outputNodes;

  private immutable uint nInputs;
  private immutable uint nOutputs;
  private immutable uint numParameters;

  // Only initialize these if training
  private double[] backPropResults = null;
  private double[] flatParms = null;
  private double[] nonBiasParms = null;

  public this(uint nInputs, uint nOutputs)
  {

    static if(is(OAF == sigmoidAF))
    {
      enforce(nOutputs == 1,"Only 1 output allowed for 2-class linear " ~
        "classification network. This error was generated to ensure the " ~
        "proper functioning of backpropagation.");
    }
    static if(is(OAF == softmaxAF))
    {
      enforce(nOutputs > 1,"More than 1 output required for general linear " ~
        "classification network. This error was generated to ensure the " ~
        "proper functioning of backpropagation.");
    }
    else
    {
      enforce(nOutputs > 0,"Number of output nodes must be greater than 1!");
    }
    
    this.nInputs = nInputs;
    this.nOutputs = nOutputs;
    this.numParameters = this.nOutputs * ( 1 + this.nInputs);

    this.inputNodes = new double[](nInputs);
    this.outputActivationNodes = new double[](nOutputs);
    static if(!is(OAF == linearAF)) this.outputNodes = new double[](nOutputs);

    this.biases = new double[](nOutputs);
    this.weights = new double[](nOutputs * nInputs);
    
    // Initialize weights and biases to random values.
    this.setRandom();
  }

  private this(uint nIn, uint nOut, double[] weights, double[] biases)
  {
    this.nInputs = nIn;
    this.nOutputs = nOut;
    this.numParameters = this.nOutputs * ( 1 + this.nInputs);

    // Set up nodes.
    this.inputNodes = new double[](nInputs);
    this.outputActivationNodes = new double[](nOutputs);
    static if(!is(OAF == linearAF)) this.outputNodes = new double[](nOutputs);

    // Copy in weights and biases.
    this.biases = biases.dup;
    this.weights = weights.dup;
  }

  package this(string str)
  {
    /*
     * String format, as put out by stringForm:
     * FeedForwardNetwork
     * LinearNetwork!OAF
     * nInputs = val
     * nOutputs = val
     * parameters = double,double,double,double....
     * 
     * end of file -  doesn't actually say this!
     * 
     * parameters are as returned by the parameters property
     */

    // Break up the lines
    string[] lines = split(str, regex("\n"));

    // Parse the header section
    string[] header = lines[0 .. 4];

    enforce(strip(header[0]) == "FeedForwardNetwork", 
      "Not a FeedForwardNetwork.");
    enforce(strip(header[1]) == "LinearNetwork!" ~ OAF.stringof, 
      "Not a LinearNetwork!" ~ OAF.stringof);

    this.nInputs = to!uint(header[2][10 .. $]);
    this.nOutputs = to!uint(header[3][11 .. $]);
    this.numParameters = this.nOutputs * (1 + this.nInputs);

    // Set up the nodes
    this.inputNodes = new double[](this.nInputs);
    this.outputActivationNodes = new double[](this.nOutputs);
    static if(!is(OAF == linearAF)) this.outputNodes = new double[](nOutputs);

    // Parse the weights
    string data = lines[4][13 .. $];
    double[] parms = new double[](this.numParameters);

    string[] tokens = split(data, regex(","));
    enforce(tokens.length == this.numParameters, 
      "Corrupt string, not enough parameters.");
    foreach(i; 0 .. this.numParameters) 
      parms[i] = to!double(strip(tokens[i]));

    this.biases = new double[](nOutputs);
    this.weights = new double[](nOutputs * nInputs);

    this.parameters = parms;
  }

  /**
   * Get the error derivative with respect to the weights via backpropagation.
   *
   * Perform backpropagation based on the values set in the network by the
   * last call to eval and return the error function gradient implied by the 
   * provided target values. Assumes that the output activation function and 
   * error function for training are properly matched.
   *
   * It turns out for all the linear network types, the back-propagation formula
   * is exactly the same when matched with the proper error function, regardless
   * of output activation function.
   * 
   * Params:
   * targets = the values the outputs should have been evaluated to on the last 
   *           call to eval.
   *
   * Returns: the gradient of the error function with respect to the parameters,
   *          or weights. This array is parallel to the array returned by
   *          the parameters property.
   */
  override ref const(double[]) backProp(in double[] targets)
  {
    assert(targets.length == nOutputs, 
      "targets.length doesn't equal the number of network outputs.");

    static if(is(OAF == linearAF)) alias outputNodes = outputActivationNodes;
    
    // Initialize if needed.
    if(backPropResults is null)
    {
      backPropResults = new double[numParameters];
    }

    // Reset to zero
    backPropResults[] = 0.0;
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs)
    {
      foreach(i; 0 .. nInputs)
      {
        backPropResults[j++] = -(targets[o] - outputNodes[o]) * inputNodes[i];
      }

      backPropResults[j++] = -(targets[o] - outputNodes[o]);
    }
    
    return backPropResults;
  }

  /**
   * Evaluate the network.
   *
   * Given the inputs, evaluate the network, storing the information needed
   * to later calculate a derivative using backProp.
   * 
   * Params:
   * inputs = the inputs to be evaluated.
   *
   * Returns: the network outputs.
   */
  override ref const(double[]) eval(in double[] inputs) {
    assert(inputs.length == this.inputNodes.length);

    /* Do I really need to copy, I could just use these as input ranges to 
       speed things up. For example...

       this.inputNodes = inputs

       This method should be called a lot, so not saving a copy might be worth 
       the efficiency increase. Espiecially when dealing with very large 
       networks.

       The downside of this is that the inputs array may be modified after
       the call to eval, and the network won't accurately remember its state.

       Of course, one should not assume the accuracy of the state anyway unless
       the last call on the network was eval, as other calls may update the
       weights, putting the network in an inconsistent internal state.
      */
    //this.inputNodes = inputs.dup;
    this.inputNodes = inputs;
    
    foreach(o; 0 .. nOutputs)
    {
      
      // Get the biases
      outputActivationNodes[o] = biases[o];
      
      // Add in the weighted input nodes
      auto offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        outputActivationNodes[o] += inputNodes[i] * weights[offset + i];
    }

    // Apply the output activation function.
    static if(is(OAF == linearAF))
    {
      alias outputNodes = outputActivationNodes;
    }
    else
    {
      OAF.eval(outputActivationNodes, outputNodes);
    }
    
    /* Do not worry about 
     */
    return outputNodes;
  }

  /**
   * Returns: A copy of this network.
   */
  override LinearNetwork!OAF dup()
  {
    // Check to make sure this is a copying constructor
    return new 
      LinearNetwork!OAF(this.nInputs,this.nOutputs,this.weights, this.biases);
  }

  /**
   * Params:
   * newParms = the new parameters, or weights, to use in the network,
   *            typically called in a trainer.
   *
   * Returns: the weights of the network organized as a 1-d array.
   */
  public override @property ref const(double[]) parameters()
  {
    // Initialize if needed
    if(flatParms is null)
    {
      flatParms = new double[numParameters];
    }

    // It seems there would be a better way to do this using slices that may 
    // be more efficient and is easier to code. I did it this way in order to 
    // keep the packing in a certain format so that I could use an SVD to train
    // the linear network via least squares. The way I would prefer to do it is:
    //
    // flatParms[0 .. (nInputs * nOutputs)] = weights.dup;
    // flatParms[(nInputs * nOutputs) .. $] = biases.dup;
    //
    // But I need the biases packed next to their weights, so I can treat the
    // biases with good old linear algebra later, much more efficiently.
    size_t j = 0;
    foreach(o; 0 .. nOutputs)
    {
      size_t offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        flatParms[j++] = weights[offset + i];
      flatParms[j++] = biases[o];
    }
    
    return flatParms;
  }

  /**
   * ditto
   */
  public override @property double[] parameters(double[] parms)
  {
    assert(parms.length == numParameters, 
      "Supplied array different size than number of parameters in network.");
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs)
    {
      size_t offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        weights[offset + i] = parms[j++];
      biases[o] = parms[j++];
    }
    
    return parms;
  }

  /**
   * Used by regularizations, which often should not affect the bias
   * weights.
   * 
   * Returns: the weights of the network with those corresponding to biases set 
   *          to zero.
   */
  public override @property ref const(double[]) nonBiasParameters()
  {
    // Initialize if needed
    if(nonBiasParms is null)
    {
      nonBiasParms = new double[this.numParameters];
    }
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs)
    {
      size_t offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        nonBiasParms[j++] = weights[offset + i];
      nonBiasParms[j++] = 0.0; // biases[o]; in the case of parameters above.
    }
    
    return nonBiasParms;
  }

  /**
   * Initialize the network weights to random values.
   */
  public override void setRandom()
  {
    /* This needs work to use a more robust random number generator.
       
       TODO

     */
    // Initalize weights with small random values
    long seed = -Clock.currStdTime;
    double scaleFactor = sqrt(1.0 / nInputs);
    foreach(j; 0 .. (numInputs * numOutputs))
    {
      weights[j] = gasdev(seed) * scaleFactor;
      if(j < nOutputs) biases[j] = gasdev(seed) * scaleFactor;
    }
  }

  /**
   * The number of inputs for the network.
   */
  public @property uint numInputs()
  {
    return nInputs;
  }

  /**
   * The number of outputs for the network.
   */
  public @property uint numOutputs()
  {
    return nOutputs;
  }

  /**
   * Returns: The weights, biases, and configuration of the network as
   *          a string that can be saved to a file.
   */
  public @property string stringForm()
  {
    /*
     * File format:
     * FeedForwardNetwork
     * LinearNetwork!OAF
     * nInputs = val
     * nOutputs = val
     * parameters = double,double,double,double....
     * 
     * end of file -  doesn't actually say this!
     * 
     * parameters are as returned by the parameters property
     */
    // Add headers
    string toRet = "FeedForwardNetwork\n";
    toRet ~= "LinearNetwork!" ~ OAF.stringof ~ "\n";
    toRet ~= format("nInputs = %d\nnOutputs = %d\n", nInputs, nOutputs);
    // Save parameters
    toRet ~= "parameters = ";
    foreach(parm; parameters) toRet ~= format("%.*e,", double.dig, parm);
    toRet = toRet[0 .. $ - 1] ~ "\n"; // Replace last comma with new-line

    return toRet;
  }
}

/**
 * Linear Regression Network.
 */
alias LinRegNet = LinearNetwork!linearAF;

/**
 * Linear Classification Network. 
 *
 * 2 classes only, 1 output only, 0-1 coding to tell the difference between
 * classes.
 */
alias Lin2ClsNet = LinearNetwork!sigmoidAF;

/**
* Linear Classification Network.
* 
* Any number of classes, but must have at least 2 outputs. Uses 1 of N coding
* on ouput nodes.
*/
alias LinClsNet = LinearNetwork!softmaxAF;

unittest
{
  mixin(announceTest("LinRegNet eval(double)"));

  // Make a fake data set
  double[][] fakeData = [[  1.0,   2.0,   3.0,   4.0,  35.0,  31.0],
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
  alias immutable(DataPoint!(numIn, numOut)) DP;
  
  // Make a data set
  iData d1 = new iData(fakeData, binaryFlags, normalize);

  // Now, build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinRegNet slprn = new LinRegNet(numIn,numOut);
  slprn.parameters = wts;

  // Now, lets test some numbers
  foreach(dp; d1.simpleRange)
    assert(approxEqual(slprn.eval(dp.inputs), dp.targets));
}


unittest{
  mixin(announceTest("LinRegNet stringForm and this(string)"));
  
  // Number of inputs and outputs
  enum numIn = 2;
  enum numOut = 1;

  // Now, build a network.
  LinRegNet slpcn = new LinRegNet(numIn,numOut);
  LinRegNet loaded = new LinRegNet(slpcn.stringForm);

  write(LinRegNet.stringof,"....");

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(approxEqual(slpcn.weights, loaded.weights));
  assert(approxEqual(slpcn.biases, loaded.biases));
}


unittest{
  mixin(announceTest("Lin2ClsNet eval(double)"));

  // Make a fake data set, this is an AND network
  double[][] andDataArr = [
      [ 0.0, 0.0, 0.0 ],
      [ 0.0, 1.0, 0.0 ],
      [ 1.0, 0.0, 0.0 ],
      [ 1.0, 1.0, 1.0 ]
      ];
  
  // Make a fake data set, this is an OR network
  double[][] orDataArr = [
      [ 0.0, 0.0, 0.0 ],
      [ 0.0, 1.0, 1.0 ],
      [ 1.0, 0.0, 1.0 ],
      [ 1.0, 1.0, 1.0 ]
  ];

  // All binary flags are true, because all of the data is binary!
  bool[] binaryFlags = [true , true, true];
  
  // Number of inputs and outputs
  enum numIn = 2;
  enum numOut = 1;
  
  // Normalize the data set (NO!, the predetermined weights for this data set
  // don't allow it.)
  enum normalize = false;

  // short hand for dealing with data
  alias immutable(Data!(numIn, numOut)) iData;
  alias immutable(DataPoint!(numIn, numOut)) DP;
  
  // Make a data set
  iData d1 = new iData(andDataArr, binaryFlags, normalize);

  // Now, build a network.
  double[] wts = [1000.0, 1000.0, -1500.0];
  Lin2ClsNet slpcn = new Lin2ClsNet(numIn,numOut);
  slpcn.parameters = wts;

  // Now, lets test some numbers
  foreach(dp; d1.simpleRange)
    assert(approxEqual(slpcn.eval(dp.inputs), dp.targets),
      format("%s == %s", slpcn.eval(dp.inputs), dp.targets));

  // Make a data set
  iData d2 = new iData(orDataArr, binaryFlags, normalize);

  // Now, build a network.
  wts = [1000.0, 1000.0, -500.0];
  slpcn.parameters = wts;

  // Now, lets test some numbers
  foreach(dp; d2.simpleRange)
    assert(approxEqual(slpcn.eval(dp.inputs), dp.targets));
}

unittest{
  mixin(announceTest("Lin2ClsNet stringForm and this(string)"));
  
  // Number of inputs and outputs
  enum numIn = 2;
  enum numOut = 1;

  // Now, build a network.
  Lin2ClsNet slpcn = new Lin2ClsNet(numIn,numOut);
  Lin2ClsNet loaded = new Lin2ClsNet(slpcn.stringForm);

  write(Lin2ClsNet.stringof,"....");

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(approxEqual(slpcn.weights, loaded.weights));
  assert(approxEqual(slpcn.biases, loaded.biases));
}

unittest{
  mixin(announceTest("LinClsNet stringForm and this(string)"));
  
  // Number of inputs and outputs
  enum numIn = 4;
  enum numOut = 3;

  // Now, build a network.
  LinClsNet slpcn = new LinClsNet(numIn,numOut);
  LinClsNet loaded = new LinClsNet(slpcn.stringForm);

  write(LinClsNet.stringof,"....");

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(approxEqual(slpcn.weights, loaded.weights));
  assert(approxEqual(slpcn.biases, loaded.biases));
}
