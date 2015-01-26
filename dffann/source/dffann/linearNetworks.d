/**
 * Collection of linear networks. 
 *
 * Author: Ryan Leach
 */
module dffann.linearNetworks;

public import dffann.dffann;

import dffann.feedforwardnetwork;

import numeric.random: gasdev;

import std.conv;
import std.datetime;
import std.exception;
import std.math;
import std.regex;
import std.string;

version(unittest){
  import dffann.data;
}

/**
 * Linear Regression Network.
 * 
 * This network is implemented with a c=Ax+b matrix type algorithm where x 
 * is the input vector, A is the weight matrix, and b is the bias vector. 
 *
 * I didn't use the Matrix type from the numeric package because I need to
 * unpack and pack up the parameters a bunch, and this would involve a lot of
 * array copying, when it is simpler and faster to just do the multiplications
 * in the eval statement here. For this I expect the network sizes to be small
 * enough so as to not benefit from any potential speed ups of parallelization
 * that can be achieved with the numeric library par version option.
 *
 */
public class LinearNetwork : feedforwardnetwork {
  
  private double[] weights;
  private double[] biases;
  private double[] inputNodes;
  private double[] outputNodes;
  private immutable uint nInputs;
  private immutable uint nOutputs;
  private immutable uint numParameters;

  public this(uint nInputs, uint nOutputs){
    
    this.nInputs = nInputs;
    this.nOutputs = nOutputs;
    this.numParameters = this.nOutputs * ( 1 + this.nInputs);

    this.inputNodes = new double[](nInputs);
    this.outputNodes = new double[](nOutputs);
    this.biases = new double[](nOutputs);
    this.weights = new double[](nOutputs * nInputs);
    
    // Initialize weights and biases to random values.
    this.setRandom();
  }

  private this(uint nIn, uint nOut, double[] weights, double[] biases){
    this.nInputs = nInputs;
    this.nOutputs = nOutputs;
    this.numParameters = this.nOutputs * ( 1 + this.nInputs);

    this.inputNodes = new double[](nInputs);
    this.outputNodes = new double[](nOutputs);
    this.biases = biases.dup;
    this.weights = weights.dup;
  }

  package this(string str){
    /*
     * String format, as put out by stringForm:
     * FeedForwardNetwork
     * LinearNetwork
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
    enforce(strip(header[1]) == "LinearNetwork", 
      "Not a LinearNetwork.");

    this.nInputs = to!uint(header[2][10 .. $]);
    this.nOutputs = to!uint(header[3][11 .. $]);
    this.numParameters = this.nOutputs * (1 + this.nInputs);

    // Set up the nodes
    this.inputNodes = new double[](this.nInputs);
    this.outputNodes = new double[](this.nOutputs);

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
   * The proper matching of error function to output activation function for
   * this network is Linear Activation Function and the Mean Squared Error.
   * 
   * Params:
   * targets = the values the outputs should have evaluated to on the last call 
   *           to eval.
   *
   * Returns: the gradient of the error function with respect to the parameters,
   *          or weights. This array is parallel to the array returned by
   *          the parameters property.
   */
  override double[] backProp(in double[] targets) {
    assert(targets.length == nOutputs, 
      "targets.length doesn't equal the number of network outputs.");
    
    double[] toRet = new double[numParameters];
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs){
      foreach(i; 0 .. nInputs)
        toRet[j++] = -(targets[o] - outputNodes[o]) * inputNodes[i];
      toRet[j++] = -(targets[o] - outputNodes[o]);
    }
    
    return toRet;
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
  override double[] eval(in double[] inputs) {
    assert(inputs.length == this.inputNodes.length);
    this.inputNodes = inputs.dup;
    
    foreach(o; 0 .. nOutputs){
      
      // Get the biases
      outputNodes[o] = biases[o];
      
      // Add in the weighted input nodes
      auto offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        outputNodes[o] += inputNodes[i] * weights[offset + i];
    }
    
    return outputNodes.dup;
  }

  /**
   * Returns: A copy of this network.
   */
  override LinearNetwork dup(){
    return new 
      LinearNetwork(this.nInputs,this.nOutputs,this.weights, this.biases);
  }

  /**
   * Params:
   * newParms = the new parameters, or weights, to use in the network,
   *            typically called in a trainer.
   *
   * Returns: the weights of the network organized as a 1-d array.
   */
  override @property double[] parameters() {
    double[] toRet = new double[numParameters];

    // It seems there would be a better way to do this using slices, that may 
    // be more efficient, but is surely easier to code. I did it this we in
    // order to keep the packing in a certain format so that I could use an 
    // SVD to train the linear network via least squares. The way I would prefer
    // to do it is...
    //
    // toRet[0 .. (nInputs * nOutputs)] = weights.dup;
    // toRet[(nInputs * nOutputs) .. $] = biases.dup;
    //
    // But I need the biases packed next to their weights, so I can treat the
    // biases with good old linear algebra later, much more efficiently.
    size_t j = 0;
    foreach(o; 0 .. nOutputs){
      size_t offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        toRet[j++] = weights[offset + i];
      toRet[j++] = biases[o];
    }
    
    return toRet;
  }

  /**
   * ditto
   */
  override @property double[] parameters(double[] parms) {
    assert(parms.length == numParameters, 
      "Supplied array different size than number of parameters in network.");
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs){
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
  override @property double[] nonBiasParameters() {
    double[] toRet = new double[this.numParameters];
    
    size_t j = 0;
    foreach(o; 0 .. nOutputs){
      size_t offset = o * nInputs;
      foreach(i; 0 .. nInputs)
        toRet[j++] = weights[offset + i];
      toRet[j++] = 0.0; // biases[o]; in the case of parameters above.
    }
    
    return toRet;
  }

  /**
   * Initialize the network weights to random values.
   */
  override void setRandom() {
    // Initalize weights with small random values
    long seed = -Clock.currStdTime;
    double scaleFactor = sqrt(1.0 / nInputs);
    foreach(j; 0 .. (numInputs * numOutputs)){
      weights[j] = gasdev(seed) * scaleFactor;
      if(j < nOutputs) biases[j] = gasdev(seed) * scaleFactor;
    }
  }

  /**
   * The number of inputs for the network.
   */
  @property uint numInputs(){
    return nInputs;
  }

  /**
   * The number of outputs for the network.
   */
  @property uint numOutputs(){
    return nOutputs;
  }

  /**
   * Returns: The weights, biases, and configuration of the network as
   *          a string that can be saved to a file.
   */
  @property string stringForm(){
    /*
     * File format:
     * FeedForwardNetwork
     * LinearNetwork
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
     toRet ~= "LinearNetwork\n";
     toRet ~= format("nInputs = %d\nnOutputs = %d\n", nInputs, nOutputs);
     // Save parameters
     toRet ~= "parameters = ";
     foreach(parm; parameters) toRet ~= format("%.*e,", double.dig, parm);
     toRet = toRet[0 .. $ - 1] ~ "\n"; // Replace last comma with new-line

     return toRet;
  }
}
/*==============================================================================
 *                     Unit Tests for LinearNetwork 
 *============================================================================*/
unittest{
  mixin(announceTest("LinearNetwork stringForm and this(string)"));
  
  // Number of inputs and outputs
  enum numIn = 4;
  enum numOut = 2;

  // Now, build a network.
  LinearNetwork slprn = new LinearNetwork(numIn,numOut);
  LinearNetwork loaded = new LinearNetwork(slprn.stringForm);

  // Test that they are indeed the same.
  assert(slprn.numParameters == loaded.numParameters);
  assert(slprn.numInputs == loaded.numInputs);
  assert(slprn.numOutputs == loaded.numOutputs);
  assert(approxEqual(slprn.weights, loaded.weights));
  assert(approxEqual(slprn.biases, loaded.biases));
    
}
unittest{
  mixin(announceTest("LinearNetwork eval(double)"));

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
  
  // Normalize the data set (NO!, the predetermined weights for this data set don't allow it.)
  enum normalize = false;

  // short hand for dealing with data
  alias immutable(Data!(numIn, numOut)) iData;
  alias immutable(DataPoint!(numIn, numOut)) DP;
  
  // Make a data set
  iData d1 = new iData(fakeData, binaryFlags, normalize);

  // Now, build a network.
  double[] wts = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
  LinearNetwork slprn = new LinearNetwork(numIn,numOut);
  slprn.parameters = wts;

  // Now, lets test some numbers
  foreach(dp; d1.simpleRange){
    assert(approxEqual(slprn.eval(dp.inputs), dp.targets));
  }

}
