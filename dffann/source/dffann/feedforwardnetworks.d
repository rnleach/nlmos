/**
 * Interface for Feed Forward Artificial Neural Networks.
 *
 * Author: Ryan Leach
*/
module dffann.feedforwardnetworks;

// TODO documentation
interface feedforwardnetworks{
	
	void eval(const double[] input, ref double[] output); // For reusing mem
	
	final double[] eval(in double[] input){
		double[] output = new double[numOutputs];
		eval(input,output);
		return output;
	}
	
	@property int numInputs();
	
	@property int numOutputs();
	
	void resetDeltas();
	
	void applyDeltas();
	
	void batchBackProp(in double[] input, in double[] targets);
	
}
