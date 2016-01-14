/**
 * Collection of linear networks. 
 *
 * Author: Ryan Leach
 */
module dffann.multilayerperceptrons;

import dffann.activationfunctions;
import dffann.feedforwardnetwork;

import numeric.random: gasdev;

import std.array;
import std.conv;
import std.datetime;
import std.exception;
import std.math;
import std.regex;
import std.string;

version(unittest) import dffann.data;

/**
 * Multilayer Perceptron.
 *
 * Params:
 * HAF = hidden activation function.
 * OAF = output activation function.
 *
 * See_Also: dffann.activationfunctions
 */
public class MultiLayerPerceptronNetwork(HAF, OAF) : FeedForwardNetwork 
if(isAF!HAF && isOAF!OAF)
{
  /*
   * Consider rewriting this class to use a flat, single array layout for the
   * weights and biases, then they could be passed around as ranges instead of
   * constantly copied around.
   *
   * Indicies:
   * l - layers. l=0 is the input layer for nodes, and the weights/biases
   *     coming FROM that layer
   * o - output. Index the output nodes, or the next layer up in the hidden
   *             nodes. 
   * i - input. Index the input nodes, or the next layer down in the hidden
   *            nodes.
   *            
   * nodes[l][i] have W[l][o][i] and B[l][o] going to nodes[l + 1][o]
   */

  private double[][][] W;
  private double[][] B;
  private double[][] nodes;
  private double[][] activations;

  private immutable uint nInputs;
  private immutable uint nOutputs;
  private immutable uint nLayers;
  private immutable uint numParameters;
  private immutable uint[] nNodes;

  // Only initialize these if training
  /* These are kept around so the memory for these arrays need not be 
     reallocated with every call to methods that use them, e.g. backProp.

     This can be a time saver in the case of methods called frequently during
     training, like backProp.
   */
  private double[] backPropResults = null;
  private double[] flatParms = null;
  private double[] nonBiasParms = null;
  private double[][][] deltaW = null;   // Used in backProp
  private double[][] deltaB = null;     // Used in backProp
  private double[][] deltaNodes = null; // Used in backProp

  /**
   * Params:
   * numNodes = description of the structure of the network. numNodes[0] is the
   *            number of input nodes. numNodes[$-1] is the number of output
   *            nodes. Everything in between is the number of hidden nodes in
   *            each hidden layer.
   */
  public this(in uint[] numNodes)
  {

    static if(is(OAF == SigmoidAF))
    {
      enforce(numNodes[$ - 1] == 1,"Only 1 output allowed for 2-class linear " ~
        "classification network. This error was generated to ensure the " ~
        "proper functioning of backpropagation.");
    }
    static if(is(OAF == SoftmaxAF))
    {
      enforce(numNodes[$ - 1] > 1,"More than 1 output required for general linear " ~
        "classification network. This error was generated to ensure the " ~
        "proper functioning of backpropagation.");
    }
    else
    {
      enforce(numNodes[$ - 1] > 0,"Number of output nodes must be greater than 0!");
    }

    enforce(numNodes.length > 2, "Must have more than 2 layers. For 2 layers" ~
      " use a linear network.");
    
    this.nNodes = numNodes.idup;
    this.nInputs = nNodes[0];
    this.nOutputs = nNodes[$ - 1];
    this.nLayers = cast(uint)numNodes.length;

    // Count the number of parameters
    // This could be calculated in a simpler way, but I like to keep this code
    // around as a simple template for iterating over the data structure.
    uint count = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. this.nNodes[l + 1])
      {
        foreach(i; 0 .. this.nNodes[l])
        {
          ++count;
        }
        ++count; // For the bias!
      }
    }
    this.numParameters = count;
    
    // Initialize the nodes
    this.nodes = new double[][nLayers];
    this.activations = new double[][nLayers];
    foreach(l; 0 .. nLayers)
    {
      this.nodes[l] = new double[nNodes[l]];
      this.activations[l] = new double[nNodes[l]];
    }
    
    // Initialize the weights and biases
    this.W = new double[][][nLayers - 1];
    this.B = new double[][nLayers - 1];
    foreach(l; 0 .. (nLayers - 1))
    {
      this.W[l] = new double[][](nNodes[l + 1], nNodes[l]);
      this.B[l] = new double[](nNodes[l + 1]);
    }
    
    // Initialize to random parameters
    this.setRandom();
  }

  private this(in uint[] numNodes, in double[][][] weights, in double[][] biases)
  {

    this.nNodes = numNodes.idup;
    this.nInputs = nNodes[0];
    this.nOutputs = nNodes[$ - 1];
    this.nLayers = cast(uint)numNodes.length;

    // Count the number of parameters
    // This could be calculated in a simpler way, but I like to keep this code
    // around as a simple template for iterating over the data structure.
    uint count = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. this.nNodes[l + 1])
      {
        foreach(i; 0 .. this.nNodes[l])
        {
          ++count;
        }
        ++count; // For the bias!
      }
    }
    this.numParameters = count;
    
    // Initialize the nodes
    this.nodes = new double[][nLayers];
    this.activations = new double[][nLayers];
    foreach(l; 0 .. nLayers)
    {
      this.nodes[l] = new double[nNodes[l]];
      this.activations[l] = new double[nNodes[l]];
    }
    
    // Initialize the weights and biases
    this.W = new double[][][nLayers - 1];
    this.B = new double[][nLayers - 1];
    foreach(l; 0 .. (nLayers - 1))
    {
      this.W[l] = new double[][nNodes[l + 1]];
      foreach(o; 0 .. nNodes[l + 1]) 
        this.W[l][o] = weights[l][o].dup;
      this.B[l] = biases[l].dup;
    }
  }

  package this(in string str)
  {
    /*
     * File format:
     * FeedForwardNetwork
     * MultiLayerPerceptronNetwork!(HAF,OAF)
     * nInputs = val
     * nOutputs = val
     * nLayers = val
     * nNodes = val,val,val,...
     * parameters = double,double,double,double....
     * 
     * end of file -  doesn't actually say this!
     * 
     * parameters are as returned by the parameters property
     */

    // Break up the lines
    string[] lines = split(str, regex("\n"));

    // Parse the header section
    string[] header = lines[0 .. 6];

    enforce(strip(header[0]) == "FeedForwardNetwork", 
      "Not a FeedForwardNetwork.");
    enforce(strip(header[1]) == "MultiLayerPerceptronNetwork!(" ~ HAF.stringof ~ 
      "," ~ OAF.stringof ~ ")", 
        "Not a MultiLayerPerceptronNetwork!(" ~ HAF.stringof ~ 
        "," ~ OAF.stringof ~ ")" ~ OAF.stringof);

    this.nInputs = to!uint(header[2][10 .. $]);
    this.nOutputs = to!uint(header[3][11 .. $]);
    this.nLayers = to!uint(header[4][10 .. $]);
    
    // Parse nNodes
    uint[] temp = new uint[nLayers];
    string[] tokens = split(header[5][9 .. $], regex(","));
    enforce(tokens.length == nLayers, format( 
      "Corrupt string, nNodes isn't nLayers in length.\n" ~ 
      " nLayers %d != tokens.length %d header[5] = %s",
      nLayers, tokens.length, header[5]));
    foreach(j; 0 .. nLayers) temp[j] = to!uint(strip(tokens[j]));
    this.nNodes = temp.idup;

    // Count the number of parameters
    // This could be calculated in a simpler way, but I like to keep this code
    // around as a simple template for iterating over the data structure.
    uint count = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. this.nNodes[l + 1])
      {
        foreach(i; 0 .. this.nNodes[l])
        {
          ++count;
        }
        ++count; // For the bias!
      }
    }
    this.numParameters = count;
    
    // Initialize the nodes
    this.nodes = new double[][nLayers];
    this.activations = new double[][nLayers];
    foreach(l; 0 .. nLayers)
    {
      this.nodes[l] = new double[nNodes[l]];
      this.activations[l] = new double[nNodes[l]];
    }

    // Parse the parameters
    string data = lines[6][13 .. $];
    double[] parms = new double[](this.numParameters);

    tokens = split(data, regex(","));
    enforce(tokens.length == this.numParameters, 
      "Corrupt string, not enough parameters.");
    foreach(i; 0 .. this.numParameters) 
      parms[i] = to!double(strip(tokens[i]));

    // Initialize the weights and biases
    this.W = new double[][][nLayers - 1];
    this.B = new double[][nLayers - 1];
    foreach(l; 0 .. (nLayers - 1))
    {
      this.W[l] = new double[][](nNodes[l + 1], nNodes[l]);
      this.B[l] = new double[nNodes[l + 1]];
    }

    // Copy in the loaded parameters.
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
   * It turns out for all the network types, the back-propagation formula
   * is exactly the same when matched with the proper error function, regardless
   * of output activation function, the only differences relate to the hidden
   * activation function.
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

    // Set up variables for calculating gradient, allocate arrays if needed.
    if(deltaW is null)
    {
      deltaW = new double[][][nLayers - 1];
      deltaB = new double[][nLayers - 1];
      deltaNodes = new double[][nLayers];

      foreach(l; 0 .. nLayers)
      {
        deltaNodes[l] = uninitializedArray!(double[])(nNodes[l]);
      }

      foreach(l; 0 .. (nLayers - 1))
      {
        deltaW[l] = uninitializedArray!(double[][])(nNodes[l + 1], nNodes[l]);
        deltaB[l] = uninitializedArray!(double[])(nNodes[l + 1]);
      }
    }

    // Initialize arrays to zeros
    foreach(l; 0 .. nLayers)
    {
      deltaNodes[l][] = 0.0;
    }

    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        deltaW[l][o][] = 0.0;
      } 
        
      deltaB[l][] = 0.0;
    }
    
    // Calculate the deltas for the output layer
    foreach(o; 0 .. nNodes[$ - 1])
      deltaNodes[$ - 1][o] = nodes[$ - 1][o] - targets[o];

    // Calculate the deltas for the rest of the layers - back propagate!
    for(auto l = nLayers - 2; l > 0; --l)
    {
      // Get the derivative of the activation function
      double[] drv = HAF.deriv(activations[l], nodes[l]);

      foreach(i; 0 .. nNodes[l])
      {
        foreach(o; 0 .. nNodes[l + 1])
          deltaNodes[l][i] += deltaNodes[l + 1][o] * W[l][o][i];
        deltaNodes[l][i] *= drv[i];
      }
    }

    // Calculate dW for the layers
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        foreach(i; 0 .. nNodes[l])
          deltaW[l][o][i] += deltaNodes[l+ 1][o] * nodes[l][i];

        deltaB[l][o] += deltaNodes[l + 1][o];
      }
    }

    flattenParms(deltaW, deltaB, backPropResults);

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
  override ref const(double[]) eval(in double[] inputs)
  {
    assert(inputs.length == this.nodes[0].length);

    // Copy inputs into the nodes
    /* Do I really need to copy, I could just use these as input ranges to 
       speed things up. For example...

       nodes[0] = inputs

       This method should be called a lot, so not saving a copy might be worth 
       the efficiency increase. Espiecially when dealing with very large 
       networks.

       The downside of this is that the inputs array may be modified after
       the call to eval, and the network won't accurately remember its state.

       Of course, one should not assume the accuracy of the state anyway unless
       the last call on the network was eval, as other calls may update the
       weights, putting the network in an inconsistent internal state.

       ANSWER: Yes, I need to copy. Because I "cannot implicitly convert 
       expression (inputs) of type const(double[]) to double[]" according to the
       compiler. The only other options would be to force some sort of 
       const-ness on nodes, but then I couldn't change them in the hidden 
       layers.
    */
    nodes[0][] = inputs[];

    // Fill the hidden layers
    foreach(l; 0 .. (nLayers - 2))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        activations[l + 1][o] = 0.0;
        foreach(i; 0 .. nNodes[l])
        {
          activations[l + 1][o] += nodes[l][i] * W[l][o][i];
        }
        
        activations[l + 1][o] += B[l][o]; // For the bias!
      }
      HAF.eval(activations[l + 1], nodes[l + 1]);
    }
    
    // Fill the output layer
    size_t idx = nLayers - 1;
    foreach(o; 0 .. nNodes[idx])
    {
      activations[idx][o] = 0.0;
      foreach(i; 0 .. nNodes[idx - 1])
      {
        activations[idx][o] += W[idx - 1][o][i] * nodes[idx - 1][i];
      }

      activations[idx][o] += B[idx - 1][o];
    }
    OAF.eval(activations[idx], nodes[idx]);
    
    return nodes[idx];  
  }

  /**
   * Returns: A copy of this network.
   */
  override MultiLayerPerceptronNetwork!(HAF, OAF) dup()
  {
    // Check constructor code to ensure it is a copying constructor.
    return new 
      MultiLayerPerceptronNetwork!(HAF, OAF)(this.nNodes, this.W, this.B);
  }
  
  
  /**
   * Make a private version of this to use for flattening the gradients
   * to return for the backProp method.
   */
  private void flattenParms(double[][][] wgts, double[][] biases, ref double[] storage)
  {
    // Initialize if necessary
    if(storage is null)
    {
      storage = new double[numParameters];
    }

    size_t p = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        foreach(i; 0 .. nNodes[l])
        {
          storage[p++] = wgts[l][o][i];
        }
        storage[p++] = biases[l][o]; // For the bias!
      }
    }
  }

  /**
   * Params:
   * parms = the new parameters, or weights, to use in the network,
   *         typically called in a trainer.
   *
   * Returns: the weights of the network organized as a 1-d array.
   */
  override @property ref const(double[]) parameters()
  {
    flattenParms(W, B, flatParms);
    return flatParms;
  }

  /**
   * ditto
   */
  override @property void parameters(const (double[]) parms)
  {
    assert(parms.length == numParameters, 
      "Supplied array different size than number of parameters in network.");

    size_t p = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        foreach(i; 0 .. nNodes[l])
        {
          W[l][o][i] = parms[p++];
        }
        B[l][o] = parms[p++]; // For the bias!
      }
    }
  }

  /**
   * Used by regularizations, which often should not affect the bias
   * weights.
   * 
   * Returns: the weights of the network with those corresponding to biases set 
   *          to zero.
   */
  override @property ref const(double[]) nonBiasParameters()
  {
    // Initialize arrays if needed
    if(nonBiasParms is null)
    {
      nonBiasParms = new double[numParameters];
    }

    size_t p = 0;
    foreach(l; 0 .. (nLayers - 1))
    {
      foreach(o; 0 .. nNodes[l + 1])
      {
        foreach(i; 0 .. nNodes[l])
        {
          nonBiasParms[p++] = W[l][o][i];
        }
        nonBiasParms[p++] = 0.0; // For the bias!
      }
    }

    return nonBiasParms;
  }

  /**
   * Initialize the network weights to random values.
   */
  override void setRandom()
  {

    // Iterate through the layers
    foreach(l; 0 .. (nLayers - 1))
    {
      // Set the scale of the random weights in this layer
      // to the square root of the reciprocal of the number of nodes in 
      // the layer the weights are coming out of.
      const double scaleFactor = sqrt(1.0 / nNodes[l]);
      foreach(o; 0 .. nNodes[l + 1])
      {
        foreach(i; 0 .. nNodes[l])
          W[l][o][i] = gasdev() * scaleFactor;
        
        B[l][o] = gasdev() * scaleFactor; // For the bias!
      }
    }
  }

  /**
   * The number of inputs for the network.
   */
  @property uint numInputs()
  {
    return nInputs;
  }

  /**
   * The number of outputs for the network.
   */
  @property uint numOutputs()
  {
    return nOutputs;
  }

  /**
   * Returns: The weights, biases, and configuration of the network as
   *          a string that can be saved to a file.
   */
  @property string stringForm()
  {
    /*
     * File format:
     * FeedForwardNetwork
     * MultiLayerPerceptronNetwork!(HAF,OAF)
     * nInputs = val
     * nOutputs = val
     * nLayers = val
     * nNodes = val,val,val,...
     * parameters = double,double,double,double....
     * 
     * end of file -  doesn't actually say this!
     * 
     * parameters are as returned by the parameters property
     */
    // Add headers
    string toRet = "FeedForwardNetwork\n";
    toRet ~= "MultiLayerPerceptronNetwork!(" ~ HAF.stringof ~ 
      "," ~ OAF.stringof ~ ")\n";
    toRet ~= format("nInputs = %d\nnOutputs = %d\nnLayers = %d\n", 
      nInputs, nOutputs, nLayers);
    toRet ~= "nNodes = ";
    foreach(j; 0 .. nLayers) toRet ~= format("%d,", nNodes[j]);
    toRet = toRet[0 .. $ - 1] ~ "\n";  // Trim off trailing comma
    // Save parameters
    toRet ~= "parameters = ";
    foreach(parm; this.parameters) toRet ~= format("%.*e,", double.dig, parm);
    toRet = toRet[0 .. $ - 1] ~ "\n"; // Replace last comma with new-line

    return toRet;
  }
}

/**
 * MLP tanh Regression Network.
 */
alias MLPRegNet = MultiLayerPerceptronNetwork!(TanhAF, LinearAF);

/**
 * MLP tanh Classification Network. 
 *
 * 2 classes only, 1 output only, 0-1 coding to tell the difference between
 * classes.
 */
alias MLP2ClsNet = MultiLayerPerceptronNetwork!(TanhAF, SigmoidAF);

/**
 * MLP tanh Classification Network.
 * 
 * Any number of classes, but must have at least 2 outputs. Uses 1 of N coding
 * on ouput nodes.
 */
alias MLPClsNet = MultiLayerPerceptronNetwork!(TanhAF, SoftmaxAF);

unittest
{
  // MLP2ClsNet eval(double)

  // Make a fake data set XOR
  double[][] fakeData = [[ 0.0, 0.0, 0.0],
                         [ 0.0, 1.0, 1.0],
                         [ 1.0, 0.0, 1.0],
                         [ 0.0, 0.0, 0.0]];

  // All binary flags are true, because all of the data is binary!
  bool[] binaryFlags = [true, true, true];
  
  // Number of inputs and outputs
  enum uint numIn = 2;
  enum uint numOut = 1;
  enum uint[] numNodes = [numIn, 2, numOut];
  
  // Normalize the data set (NO!, the predetermined weights for this data set
  // don't allow it.)
  enum normalize = false;

  // short hand for dealing with data
  alias DataType = Data!(numIn, numOut);
  alias iData = immutable(Data!(numIn, numOut));
  alias DP =immutable(DataPoint!(numIn, numOut));
  
  // Make a data set
  iData d1 = DataType.createImmutableData(fakeData, binaryFlags, normalize);


  // Now, build a network.
  const double[] wts = [ 1000.0, -1000.0, -500.0, 
                        -1000.0,  1000.0, -500.0, 
                         1000.0,  1000.0, 500.0];

  MLP2ClsNet slprn = new MLP2ClsNet(numNodes);
  slprn.parameters = wts;

  // Now, lets test some numbers
  foreach(dp; d1.simpleRange)
    assert(approxEqual(slprn.eval(dp.inputs), dp.targets),
      format("eval(%s) = %s != targets = %s", 
        dp.inputs, slprn.eval(dp.inputs), dp.targets));
}

unittest
{
  // MLPRegNet stringForm and this(string)
  
  // Number of nodes per layer
  enum uint[] numNodes = [2,5,6,2];

  // Now, build a network.
  MLPRegNet slpcn = new MLPRegNet(numNodes);
  MLPRegNet loaded = new MLPRegNet(slpcn.stringForm);

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(slpcn.nLayers == loaded.nLayers);
  assert(approxEqual(slpcn.W, loaded.W));
  assert(approxEqual(slpcn.B, loaded.B));
}

unittest
{
  // MLP2ClsNet stringForm and this(string)
  
  // Number of nodes per layer
  enum uint[] numNodes = [4,5,6,1];

  // Now, build a network.
  MLP2ClsNet slpcn = new MLP2ClsNet(numNodes);
  MLP2ClsNet loaded = new MLP2ClsNet(slpcn.stringForm);

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(slpcn.nLayers == loaded.nLayers);
  assert(approxEqual(slpcn.W, loaded.W));
  assert(approxEqual(slpcn.B, loaded.B));
}

unittest
{
  // MLPClsNet stringForm and this(string)
  
  // Number of nodes per layer
  enum uint[] numNodes = [4,5,6,3];

  // Now, build a network.
  MLPClsNet slpcn = new MLPClsNet(numNodes);
  MLPClsNet loaded = new MLPClsNet(slpcn.stringForm);

  // Test that they are indeed the same.
  assert(slpcn.numParameters == loaded.numParameters);
  assert(slpcn.numInputs == loaded.numInputs);
  assert(slpcn.numOutputs == loaded.numOutputs);
  assert(slpcn.nLayers == loaded.nLayers);
  assert(approxEqual(slpcn.W, loaded.W));
  assert(approxEqual(slpcn.B, loaded.B));
}
