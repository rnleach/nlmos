/**
 * Interface for Feed Forward Artificial Neural Networks.
 *
 * Author: Ryan Leach
*/
module dffann.feedforwardnetwork;

public import dffann.dffann;

/**
 * Interface for feed foward networks.
 */
interface feedforwardnetwork{
	
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
	ref const(double[]) eval(in double[] input);
  
  /**
   * Get the error derivative with respect to the weights via backpropagation.
   *
	 * Perform backpropagation based on the values set in the network by the
	 * last call to eval and return the error function gradient implied by the 
	 * provided target values. Assumes that the output activation function and 
	 * error function for training are properly matched.
	 * 
	 * Proper matchings of error function to output activation function are
	 * outlined below:
	 * 
	 * Linear Activation Function -> Mean Squared Error
	 * Rectified Linear Activation Function -> Squared Error Function
	 * Sigmoid Activation Function -> 2-class single output Cross Entropy Error
	 * SoftmaxActivationFunction -> Cross Entropy Error Function
	 * 
	 * Params:
	 * targets = the values the outputs should have evaluated to on the last call 
	 *           to eval.
	 *
	 * Returns: the gradient of the error function with respect to the parameters,
	 *          or weights. This array is parallel to the array returned by
	 *          the parameters property.
	 */
	ref const(double[]) backProp(in double[] targets);
	
	/**
	 * The number of inputs for the network.
	 */
	@property uint numInputs();
	
	/**
	 * The number of outputs for the network.
	 */
	@property uint numOutputs();
	
	/**
	 * Returns: the weights of the network organized as a 1-d array.
	 */
	@property ref const(double[]) parameters();
  
  /**
   * Params:
	 * newParms = the new parameters, or weights, to use in the network,
	 *            typically called in a trainer.
	 */
	@property double[] parameters(double[] newParams);

  /**
	 * Used by regularizations, which often should not affect the bias
	 * weights.
	 * 
	 * Returns: the weights of the network with those corresponding to biases set 
	 *          to zero.
	 */
	@property ref const(double[]) nonBiasParameters();

	/**
	 * Initialize the network weights to random values.
	 */
	void setRandom();

	/**
	 * Returns: The weights, biases, and configuration of the network as
	 *          a string that can be saved to a file.
	 */
	@property string stringForm();

	/**
	 * Returns: A copy of this network.
	 */
	@property feedforwardnetwork dup();
	
}
