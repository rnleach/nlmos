/**
 * Utilities for loading, saving, and passing data around.
 */
module dffann.data;

import std.array;
import std.conv;
import std.exception;
import std.math;
import std.range;
import std.regex;
import std.stream;
import std.string;

/*------------------------------------------------------------------------------
 *                       Alias Types for Clarity
 *----------------------------------------------------------------------------*/
 /**
  * Data is an alias to the mData class for immutable objects only. There is no 
  * way to insantiate a mutable version of mData.
  */
alias immutable(mData) Data;
/**
 * TrainingData is a more descriptive name for immutable(double[][]) where a 
 * pair of InputData and TargetData are expected.
 */
alias immutable(double[][]) TrainingData;
/**
 * InputData is a more descriptive name for immutable(double[]) in places where
 * network inputs are expected.
 */
alias immutable(double[]) InputData;
/**
 * TargetData is a more descriptive name for immutable(double[]) in places where
 * network targets (used for training) are expected.
 */
alias immutable(double[]) TargetData;

/**
 * Used to delineate what data type is desired in templates.
 */
enum DataType {TRAINING_DATA, INPUT_DATA, TARGET_DATA};

version(unittest){

  // Set up some variables to be available for all unit tests.
  
  import std.stdio;
  import std.file;

  // Generate string for mixin that announces this test.
  string announceTest(in string msg){
    return "
    write(format(\"Testing %s - %5d: %s...\",__FILE__,__LINE__,\"" ~ msg ~"\"));
    scope(exit)writeln(\"done.\");";
  }
  
  // Test Data
  double[][] testData = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
    [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
    [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    [1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    [1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6],
    [1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7],
    [1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8],
    [1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9]];
  
  // Values for all normalized values in each row of the test data.
  double[] normalizedRowValues = 
   [-1.486301082920590,
    -1.156011953382680,
    -0.825722823844772,
    -0.495433694306863,
    -0.165144564768954,
     0.165144564768954,
     0.495433694306863,
     0.825722823844772,
     1.156011953382680,
     1.486301082920590];
  
  // None of these values are binary
  bool[] flags = [false, false, false, false, false, false, false];
  
  // Tolerance for calculations when comparing calculated values during tests.
  enum TOL = 1.0e-12;
}

/*******************************************************************************
 * mData is mutable data, but that is really a misnomer, because the class is 
 * intended to only be used in immutable form, hence the alias above to Data.
 *
 * This class manages the data and keeps data inputs paired with targets for
 * training a network. It is also possible to use this to just hold input data
 * by setting the number of targets to zero and just calling methods to get
 * inputs only.
 * 
 * Authors: Ryan Leach
 * Date: October 15, 2014
 * See_Also:
 * 
*******************************************************************************/
class mData
{
  // Data values, inputs and targets should be parallel arrays.
	private double[][] inputs;
	private double[][] targets;
	
	// Information for creating normalizations.
	private double[] inputShift;
	private double[] inputScale;
	private double[] targetShift;
	private double[] targetScale;
	
	// false values mean they are 'real-valued', true means that input/target
	// should not be filtered because it is binary (0,1) or (-1,1)
	private bool[] inputFilter;
	private bool[] targetFilter;
	
	// Number of points in the data set, number of input values and target values
	// per point.
	private size_t nPoints;
	private size_t nInputs;
	private size_t nTargets;
	
	/**
	 * Immutable constructor, only immutable instances of data are allowed. This
	 * constructor assumes the data is not normalized, and normalizes it.
	 *
	 * Params: 
   * data    = A 2-d array with rows representing a data point and columns, e.g. 
   *           1000 rows by 5 columns is 1000 5-dimensional data points. If
   *           there are 3 inputs and 2 targets, then the targets are the last
   *           two values in each point.
	 * filter  = Must be size inputs + targets. True values indicate that the 
   *           corresponding column is a binary input (this is not checked) and 
   *           has value 0 or 1. Thus it should not be normalized.
	 * inputs  = The number of columns which represent inputs.
	 * targets = The number of columns which represent targets.
	 *
	 * See_Also: Normalizations
	 */
	private immutable this(const double[][] data,
	                       const bool[] filter,
	                       const size_t inputs, 
	                       const size_t targets){
	  // Check lengths
    enforce(data.length > 1, 
      "Initialization Error, no points in supplied array.");
    
    enforce(inputs + targets  == data[0].length,
      "Initialization Error, sizes do not match.");
	  
    enforce(filter.length == data[0].length,
      "Initialization Error, filters do not match data.");
	
	  this.nPoints = data.length;
	  this.nInputs = inputs;
	  this.nTargets = targets;
	  this.inputFilter = filter[0 .. inputs].idup;
    this.targetFilter = filter[inputs .. $].idup;
	  size_t numVals = targets + inputs;

    // Set up the local data storage and copy values
    //
    // inputData and targetData have scope only in this constructor, so when
    // it exits there is no other reference to the newly copied data, except 
    // the immutable ones!
    double[][] inputData = new double[][](nPoints, nInputs);
    double[][] targetData = new double[][](nPoints, nTargets);
    for(size_t i = 0; i < nPoints; ++i){
      for(size_t j = 0; j < numVals; ++j){
        if(j < nInputs) inputData[i][j] = data[i][j];
        else targetData[i][j - nInputs] = data[i][j];
      }
    }
	  
	  // Normalize inputData and TargetData in place while calculating 
    // shifts and scales
	  double[] inputShift_tmp = new double[](nInputs);
    double[] inputScale_tmp = new double[](nInputs);
    double[] targetShift_tmp = new double[](nTargets);
    double[] targetScale_tmp = new double[](nTargets);
    normalize(inputData, inputShift_tmp, inputScale_tmp, inputFilter);
    normalize(targetData, targetShift_tmp, targetScale_tmp, targetFilter);

    this.inputs = cast(immutable) inputData;
    this.inputShift = cast(immutable)inputShift_tmp;
    this.inputScale = cast(immutable)inputScale_tmp;

    this.targets = cast(immutable) targetData;
    this.targetShift = cast(immutable)targetShift_tmp;
    this.targetScale = cast(immutable)targetScale_tmp;
	  
	}
	
	/**
	 * Immutable constructor, only immutable instances of data are allowed. This
	 * constructor assumes the data IS already normalized, and does not normalize
	 * it again.
	 * 
	 * Params: 
   * inputData    = A 2-d array with rows representing a data point and columns,
   *                e.g. 1000 rows by 5 columns is 1000 5-dimensional data 
   *                points.
   * inputFilter  = Must be size inputs.  True values indicate that the 
   *                corresponding column is a binary input (this is not checked)
   *                and has value 0 or 1. This is also used in the 
   *                normalization, so we need it again.
   * inputShift   = The shift used in the input normalization.
   * inputScale   = The scale used in the input normalization.
   * targetData   = A 2-d array with rows representing a data point and columns,
   *                e.g. 1000 rows by 5 columns is 1000 5-dimensional data 
   *                points. There must be exactly as many points as in inputData
   *                so they are matched.
   * targetFilter = Must be size targets. True values indicate that the
   *                corresponding column is a binary value (this is not checked)
   *                and has value 0 or 1. This is also used in the 
   *                normalization, so we need it again.
   * targetShift  = The shift used in the normalization.
	 * targetScale  = The scale used in the normalization.
	 *
	 * See_Also: Normalizations.
	 */
	private immutable this(const double[][] inputData, 
                         const bool[] inputFilter,
	                       const double[] inputShift,
                         const double[] inputScale,
                         const double[][] targetData,
                         const bool[] targetFilter,
                         const double[] targetShift,
                         const double[] targetScale){
    
    // Check lengths
    enforce(inputData.length > 1, 
      "Initialization Error, no points in supplied array.");
	  
    enforce(inputData.length == targetData.length, 
      "Initialization Error, sizes of input and target data arrays not equal.");
	  
    enforce(inputFilter.length == inputData[0].length, 
      "Initialization Error, input filters do not match data.");
	  
    enforce(inputShift.length == inputData[0].length, 
      "Initialization Error, input shifts do not match data.");
    
    enforce(inputScale.length == inputData[0].length, 
      "Initialization Error, input scales do not match data.");
    
    enforce(targetFilter.length == targetData[0].length, 
      "Initialization Error, target filters do not match data.");
    
    enforce(targetShift.length == targetData[0].length, 
      "Initialization Error, target shifts do not match data.");
    
    enforce(targetScale.length == targetData[0].length, 
      "Initialization Error, target scales do not match data.");
	  
    // Assign values
    this.nPoints = inputData.length;
    this.nInputs = inputData[0].length;
    this.nTargets = targetData[0].length;
    this.inputFilter = inputFilter.idup;
    this.inputScale = inputScale.idup;
    this.inputShift = inputShift.idup;
    this.targetFilter = targetFilter.idup;
    this.targetShift = targetShift.idup;
    this.targetScale = targetScale.idup;
	  size_t numVals = nInputs + nTargets;
	  
	  // tmp has scope only in this constructor, so when it exits there
	  // is no other reference to the newly copied data, except the immutable
	  // one!
	  double[][] tmpInputs = new double[][](nPoints, nInputs);
    double[][] tmpTargets = new double[][](nPoints,nTargets);
	  
	  // Copy data by value into tmp arrays
	  foreach(size_t i,const double[] d; inputData) 
	    foreach(size_t j, double dd; d) tmpInputs[i][j]= dd;
    this.inputs = cast(immutable) tmpInputs;

    foreach(size_t i,const double[] d; targetData) 
      foreach(size_t j, double dd; d) tmpTargets[i][j]= dd;
    this.targets = cast(immutable) tmpTargets;
	}
	
	/**
	 * Returns: The number of points in the Data set.
	 */
	@property public final size_t numPoints()const{return this.nPoints;}
	
	/**
	 * Returns: The number of input values.
	 */
	@property public final size_t numInputs()const{return this.nInputs;}
	
	/**
	 * Returns: The number of target values.
	 */
	@property public final size_t numTargets()const{return this.nTargets;}
	
	/**
	 * Returns: DataRange for this object that iterates over the TrainingData.
	 *
	 * See_Also: TrainingData
	 */
	@property final DataRange!(DataType.TRAINING_DATA) trainingDataRange() immutable{
	  return DataRange!(DataType.TRAINING_DATA)(this);
	}
	
	/**
	 * Returns: DataRange for this object that iterates over the InputData.
	 *
	 * See_Also: InputData
	 */
	@property final DataRange!(DataType.INPUT_DATA) inputDataRange()immutable{
	  return DataRange!(DataType.INPUT_DATA)(this);
	}
	
	/**
	 * Returns: DataRange for this object that iterates over the TargetData.
	 *
	 * See_Also: TargetData
	 */
	@property final DataRange!(DataType.TARGET_DATA) targetDataRange()immutable{
	  return DataRange!(DataType.TARGET_DATA)(this);
	}
	
	/**
	 * Get an (immutable) set of network inputs and targets, used in training
	 * and verification of networks.
	 *
	 * Params: 
   * i = the index of the data point you want, ranges from 0 to the property 
   *     numPoints. This should always return the save valuefor the given data
   *     set.
	 */
	public final TrainingData getTrainingData(size_t i)immutable{
    assert(i < this.nPoints);

	  return [this.inputs[i],this.targets[i]];
	}
	
	/**
	 * Get an (immutable) set of network inputs.
	 *
	 * Params: 
   * i = the index of the data point you want, ranges from 0 to the property 
   *     numPoints. This should always return the save valuefor the given data 
   *     set.
	 */
	public final InputData getInputData(size_t i) immutable{
    assert(i < this.nPoints);

    return this.inputs[i];
	}
	
	/**
	 * Get an (immutable) set of network targets. Should not be used often, expect
	 * to use getTrainingData during training/verification and getInputData when
	 * there are no targets.
	 *
	 * Params: 
   * i = the index of the data point you want, ranges from 0 to the property 
   *     numPoints. This should always return the save value for the given data
   *     set.
	 */
	public final TargetData getTargetData(size_t i)immutable{
    assert(i < this.nPoints);

    return this.targets[i];
	}
	
	/**
	 * Returns: Normalization for input data or training data.
	 */
	public final Normalization getNormalization(DataType dt)() immutable
	if(dt == DataType.INPUT_DATA || dt == DataType.TARGET_DATA)
	{
	  static if(dt == DataType.INPUT_DATA){
	    return Normalization(this.inputShift, this.inputScale);
	  }
	  else {
	    return Normalization(this.targetShift, this.targetScale);
	  }
	}
	 
	/**
	 * Normalizes the array dt in place, returning the shift and scale of the
	 * normalization in the so-named arrays. Filters are used to mark binary 
   * inputs and automatically set their shift to 0 and scale to 1.
   *
   * TODO parallelize this section to speed it up if possible.
	 */
	private final void normalize(double[][] dt, 
	                       double[] shift, 
	                       double[] scale,
	                       const bool[] filters) immutable
	{
	  size_t numVals = shift.length;
	  double[] sum = new double[](numVals);
	  double[] sumsq = new double[](numVals);
	  
	  // Initialize
	  shift[] = 0.0;
	  scale[] = 1.0;
	  sum[] = 0.0;
	  sumsq[] = 0.0;
	  
	  // Calculate the sum and sumsq
	  foreach(d; dt){
	    for(size_t i = 0; i < numVals; ++i){
	      if(!filters[i]){
	        sum[i] += d[i];
	        sumsq[i] += d[i] * d[i];
	      }
	    }
	  }
	  
	  // Calculate the mean (shift) and standard deviation (scale)
	  for(size_t i = 0; i < numVals; ++i){
	    if(!filters[i]){
	      shift[i] = sum[i] / nPoints;
	      scale[i] = sqrt((sumsq[i] / nPoints - shift[i] * shift[i]) * nPoints / (nPoints - 1));
	    }
	  }
	  
	  // Now use these to normalize the data
	  for(size_t i = 0; i < numPoints; ++i){
	    for(size_t j = 0; j < numVals; ++j)
	      dt[i][j] = (dt[i][j] - shift[j]) / scale[j];
	  }
	  
	  // All done, now return.
	}
}
unittest{
  mixin(announceTest("Data Test."));
  
  Data d = new Data(testData, flags, 5, 2);
  
  // Test the newly loaded objects parameters
  assert(d.numPoints == 10);
  assert(d.numInputs == 5);
  assert(d.numTargets == 2);

  // Test normalization of data points
  for(size_t i = 0; i < d.nPoints; ++i){
    for(size_t j = 0; j < d.nInputs; ++j) 
      assert(abs(d.inputs[i][j] - normalizedRowValues[i]) < TOL);
    for(size_t j = 0; j < d.nTargets; ++j) 
      assert(abs(d.targets[i][j] - normalizedRowValues[i]) < TOL);
  }
}
/*-----------------------------------------------------------------------------
 *                           Data Helper Functions
 *---------------------------------------------------------------------------*/
/**
 * Get a Data object from the provided array.
 *
 * Params: 
 * d       = The array to manage as Data. Each row is considered point
 *           or a sample.
 * numIn   = The number of values in a sample that are inputs. These 
 *           are assumed to be the first numIn values in each row.
 * numTarg = The number of values in a sample that are targets. These
 *           are assumed to be the last numTarg values in each row.
 *           numIn + numTarg must equal the length of each row.
 * filters = Sometimes inputs/targets are binary, and  you don't want 
 *           them to be normalized. For each input/target column that 
 *           is binary the corresponding value in the filters array
 *           is true. The length of the filters array must be numIn + numTarg. 
 */
Data LoadDataFromArray(const double[][] d, 
                       const size_t numIn, 
                       const size_t numTarg, 
                       const bool[] filters){
  return new Data(d, filters, numIn, numTarg);
}

unittest{
  mixin(announceTest("LoadDataFromArray Test."));

  Data d = LoadDataFromArray(testData, 5, 2, flags);
  
  assert(abs(d.getTrainingData(3)[0][2] - normalizedRowValues[3]) < TOL);
  assert(abs(d.getTrainingData(4)[1][1] - normalizedRowValues[4]) < TOL);
  assert(abs(d.getInputData(0)[0] - normalizedRowValues[0]) < TOL);
  assert(abs(d.getTargetData(1)[0] - normalizedRowValues[1]) < TOL);
}

/**
 * Load data from a file and create a Data object.
 *
 * Params: 
 * filename   = Path to the data to load
 * filters    = Sometimes inputs/targets are binary, and  you don't want 
 *              them to be normalized. For each input/target column that 
 *              is binary the corresponding value in the filters array is true.
 *              The length of the filters array must be numIn + numTarg. 
 * numInputs  = The number of values in a sample that are inputs. These 
 *              are assumed to be the first numIn values in each row.
 * numTargets = The number of values in a sample that are targets. These
 *              are assumed to be the last numTarg values in each row.
 *              numInputs + numTargets must equal the length of each row.
 */
Data LoadDataFromCSVFile(const string filename, bool[] filters, 
                         const size_t numInputs, const size_t numTargets){
  
  // Open the file
  Stream f = new BufferedFile(filename);
  scope(exit) f.close();

  // Read the file line by line
  size_t numVals = numTargets + numInputs;
  auto app = appender!(double[][])();
  auto sepRegEx = ctRegex!r",";
  foreach(ulong lm, char[] line; f){
    
    // Split the line on commas
    char[][] tokens = split(line,sepRegEx);
    
    // Ensure we have enough tokens to do the job, and that the line is
    // numeric. This should skip any kind of 'comment' line or column headers
    // that start with non-numeric strings.
    if(tokens.length != numVals || !isNumeric(tokens[0])) continue;
    
    // Parse the doubles from the strings
    double[] lineValues = new double[](numVals);
    for(size_t i = 0; i < numVals; ++i){
      lineValues[i] = to!double(tokens[i]);
    }
    
    // Add them to my array
    app.put(lineValues);
  }
  
  // Return the new Data instance
  return new Data(app.data, filters, numInputs, numTargets);
}

/** 
 * Save normalized data in a pre-determined format so that it 
 * can be loaded quickly without the need to re-calculate the normalization.
 * 
 * Params: 
 * pData    = The data object to save.
 * filename = The path to save the data.
 */
void SaveProcessedData(const mData pData, const string filename){
	/*
	 * File format:
	 * NormalizedData
	 * nPoints = val
	 * nInputs = val
	 * nTargets = val
	 * inputFilter = bool,bool,bool,bool,...
	 * inputShift = val,val,val,val,val,...
	 * inputScale = val,val,val,val,val,...
	 * targetFilter = bool,bool,...
	 * targetShift = val,val,...
	 * targetScale = val,val,...
	 * data =
	 * inputVal,inputVal,inputVal,inputVal,.....targetVal,targetVal,...
	 */

	// Open the file
	Stream fl = new BufferedFile(filename, FileMode.OutNew);
	scope(exit) fl.close();

  // Put a header to identify it as NormalizedData
  fl.writefln("NormalizedData");

  // Save the variables
  fl.writefln("nPoints = %d",pData.nPoints);
  fl.writefln("nInputs = %d",pData.nInputs);
  fl.writefln("nTargets = %d",pData.nTargets);

  // Nested function for mixin - compile time evaluated to write code
  // Params: vname  - the variable name to use in this code
  //         format - A format specifier for writing array elements, e.g. "%d"
  //         prcis - A precision specifier, use blank "" if a string
  string insertWriteArray(string vname, string format, string precis = ""){
    return "
      fl.writef(\"" ~ vname ~ " = \");
      for(int i = 0; i < pData." ~ vname ~ ".length - 1; ++i)
        fl.writef(\"" ~ format ~ ",\"" ~ precis ~ ", pData." ~ vname ~ "[i]);
      fl.writef(\"" ~ format ~ "\n\"" ~ precis ~ ", pData." ~ vname ~ "[$ - 1]);
    ";
  }

  // Save the filters and normalizations 
  mixin(insertWriteArray("inputFilter","%s"));
  mixin(insertWriteArray("inputShift","%.*f",",double.dig"));
  mixin(insertWriteArray("inputScale","%.*f",",double.dig"));

  mixin(insertWriteArray("targetFilter","%s"));
  mixin(insertWriteArray("targetShift","%.*f",",double.dig"));
  mixin(insertWriteArray("targetScale","%.*f",",double.dig"));

  // Now write the output
  fl.writefln("data = ");
  for(size_t i = 0; i < pData.nPoints; ++i){
    for(size_t j = 0; j < pData.nInputs; ++j) 
      fl.writef("%.*f,", double.dig, pData.inputs[i][j]);
    for(size_t j = 0; j < pData.nTargets - 1; ++j) 
      fl.writef("%.*f,", double.dig, pData.targets[i][j]);
    fl.writef("%.*f\n", double.dig, pData.targets[i][$ - 1]);
  }

}

/** 
 * Load data that has been pre-processed and normalized into a
 * data array quickly.
 * 
 * Params: 
 * filename = The path to the file to load.
 * 
 * Returns: An immutable Data object.
 */
Data LoadProcessedData(const string filename){

  // See comments in SaveProcessedData for file and header formats.

  // Open the file, read in the contents
  Stream fl = new BufferedFile(filename, FileMode.In);
  string text = fl.toString();
  fl.close();

  // Split into lines as header and data sections
  string[] lines = split(text, regex("\n"));
  string[] header = lines[0 .. 11];
  string[] dataLines = lines[11 .. $];

  // clean up the header lines
  foreach(ref line; header) line = strip(line);

  // Parse some variables out of the header section
  enforce(header[0] == "NormalizedData");
  size_t numPoints = to!size_t(header[1][10 .. $]);
  size_t numInputs = to!size_t(header[2][10 .. $]);
  size_t numTargets = to!size_t(header[3][11 .. $]);
  size_t totalVals = numInputs + numTargets;

  // set up some variables, define nested function for mixin to
  // parse arrays.
  string tokens[];
  // Parms: vname  - e.g. "inputFilter"
  //        elType - e.g. "bool", "double"
  //        row    - e.g. "4", row number of header array to parse.
  string insertParseArray(string vname, string elType, string row){
    string numVals;
    if(indexOf(vname,"input") > -1) numVals = "numInputs";
    else numVals = "numTargets";
    string startCol = to!string(vname.length + 3);

    return "tokens = split(header["~row~"]["~startCol~" .. $],regex(\",\"));
      "~vname~" = new "~elType~"[]("~numVals~");
      for(int i = 0; i < "~numVals~"; ++i) 
        "~vname~"[i] = to!"~elType~"(strip(tokens[i]));";
  }

  bool[] inputFilter;
  mixin(insertParseArray("inputFilter","bool","4"));

  double[] inputShift;
  mixin(insertParseArray("inputShift","double","5"));

  double[] inputScale;
  mixin(insertParseArray("inputScale","double","6"));

  bool[] targetFilter;
  mixin(insertParseArray("targetFilter","bool","7"));

  double[] targetShift;
  mixin(insertParseArray("targetShift","double","8"));

  double[] targetScale;
  mixin(insertParseArray("targetScale","double","9"));

  // Now parse each row
  enforce(dataLines.length >= numPoints,
    "Malformed data file, not enough input points.");

  double[][] tmpInputs = new double[][](numPoints,numInputs);
  double[][] tmpTargets = new double[][](numPoints,numTargets);
  for(int i = 0; i < numPoints; ++i){

    tokens = split(dataLines[i],regex(","));

    enforce(tokens.length >= totalVals,
      "Malformed data file, not enought points on line.");

    for(int j = 0; j < numInputs; ++j){
      tmpInputs[i][j] = to!double(tokens[j]);
    }

    for(int j = 0; j < numTargets; ++j){
      tmpTargets[i][j] = to!double(tokens[j + numInputs]);
    }
  }

  return new Data(tmpInputs,  inputFilter,  inputShift, inputScale,
              tmpTargets, targetFilter, targetShift, targetScale);

}

/**
 * Check if the supplied path is to a data file that has been pre-processed or
 * not.
 * 
 * Params: 
 * filename = The path to the file to check.
 * 
 * Returns: true if the file is laid out as expected for something that has been
 * saved with SaveProcessedData.
 * 
 */
bool isProcessedDataFile(const string filename){

  // Open the file
  Stream fl = new BufferedFile(filename, FileMode.In);
  scope(exit) fl.close();

  // Read the first line.
  string firstLine = cast(string)fl.readLine();

  // Test it
  return firstLine == "NormalizedData";
}

unittest{
  mixin(announceTest("File handling for data tests."));

  // Not a good unittest, it relies on an external file.
  bool[] filters2 = [false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false];
  
  assert(exists("MissoulaTempAllData.csv"));
  auto d2 = LoadDataFromCSVFile("MissoulaTempAllData.csv", filters2, 21, 1);
  
  SaveProcessedData(d2,"MissoulaTempAllDataNrm.csv");
  scope(exit) std.file.remove("MissoulaTempAllDataNrm.csv");
  assert(isProcessedDataFile("MissoulaTempAllDataNrm.csv"));
  
  auto d3 = LoadProcessedData("MissoulaTempAllDataNrm.csv");
  assert(d2.numPoints == d3.numPoints);
  assert(d2.numInputs == d3.numInputs);
  
  for(size_t k = 0; k < d3.numPoints; ++k)
    for(size_t kk = 0; kk < d3.numInputs; ++kk)
      assert(approxEqual(d3.getInputData(k)[kk], d2.getInputData(k)[kk]),
        format("%.*f != %.*f at k = %d and kk = %d", double.dig,
          d3.getInputData(k)[kk], double.dig, d2.getInputData(k)[kk], k, kk));
    
}
/*------------------------------------------------------------------------------
 *                                   DataRange
 *----------------------------------------------------------------------------*/
/**
 * InputRange for training Data objects.
 */
public struct DataRange(DataType dt){

	private immutable(size_t) length;
	private size_t next;
	private Data data;
	
	/**
   * Params: 
   * d = The Data object you wish to iterate over.
   */
	this(Data d){
	  this.length = d.numPoints;
	  this.next = 0;
	  this.data = d;
  }
	
  // Properties/methods to make this an InputRange
	@property bool empty(){return next == length;}

  @property auto front(){
    static if(dt == DataType.TRAINING_DATA)
      return this.data.getTrainingData(next);
    else static if(dt == DataType.INPUT_DATA)
      return this.data.getInputData(next);
    else return this.data.getTargetData(next);
  }

	 void popFront(){++next;}
}
static assert(isInputRange!(DataRange!(DataType.TRAINING_DATA)));
static assert(isInputRange!(DataRange!(DataType.INPUT_DATA)));
static assert(isInputRange!(DataRange!(DataType.TARGET_DATA)));

unittest{
  mixin(announceTest("DataRange Test."));

  Data d = LoadDataFromArray(testData,5 ,2, flags);

  auto r = DataRange!(DataType.TRAINING_DATA)(d);
  size_t i = 0;
  foreach(t; r) assert(t == d.getTrainingData(i++));
  auto rr = d.trainingDataRange;
  i = 0;
  foreach(t; rr) assert(t == d.getTrainingData(i++));
  
  auto rI = DataRange!(DataType.INPUT_DATA)(d);
  i = 0;
  foreach(t; rI) assert(t == d.getInputData(i++));
  
  auto rT = DataRange!(DataType.TARGET_DATA)(d);
  i = 0;
  foreach(t; rT) assert(t == d.getTargetData(i++));
}
/*------------------------------------------------------------------------------
 *                                Normalizations
 *----------------------------------------------------------------------------*/
/* Will use a struct for this, but considered using closures. I foresee 
* situations where this is called a lot, so a closure might be pretty 
* inefficient considering a struct can be allocated on the stack, whereas a
* closure would necessarily be allocated on the heap.
*/
/**
 * Shorthand for immutable normalization. Expect it to only be used this way.
 */
alias immutable(mNormalization) Normalization;

/**
 * A normalization is used to force data to have certain statistical properties,
 * such as a standard deviation of 1 and a mean of 0. 
 * 
 * Sometimes data is binary, e.g. with values 0 or 1, or possibly -1 or 1, in 
 * which case you do not want to normalize it. Binary data may also be mixed 
 * with non-binary data. This is handled at the constructor level when loading
 * the data with the use of filters to determine if a column is binary.
 */
struct mNormalization{

  private double[] shift;
  private double[] scale;
  
  /**
  * Params: 
  * shift = array of shifts to subtract from each point to be normalized.
  * scale = array of scales to divide each point by.
  */
  private immutable this(const double[] shift, const double[] scale)
  {
    enforce(shift.length == scale.length, 
      "Shift and scale array lengths differ.");

    this.shift = shift.idup;
    this.scale = scale.idup;
  }

  /**
   * Given a normalized input/output set of values, unnormalize them.
   */
  void unnormalize(double[] d) inout {
    assert(d.length == shift.length && d.length == scale.length);

    d[] = d[] * scale[] + shift[];
  }
  
  /**
   * Given an unormalized input/output set of values, normalize them.
   */
  void normalize(double[] d) inout {
    assert(d.length == shift.length && d.length == scale.length);

    d[] = (d[] - shift[]) / scale[];
  }
}
unittest{
  mixin(announceTest("Normalizations Test."));

  Data d = LoadDataFromArray(testData,5 ,2, flags);

  auto inNorm = d.getNormalization!(DataType.INPUT_DATA);
  auto outNorm = d.getNormalization!(DataType.TARGET_DATA);
  
  // Test normalizing data
  for( int i = 0; i < d.nPoints; ++i){
    double[] rawIn = testData[i][0 .. d.nInputs].dup;
    inNorm.normalize(rawIn);
    for(size_t j = 0; j < d.nInputs; ++j){
      assert(rawIn[j] == d.inputs[i][j]);
    }
    double[] rawOut = testData[i][d.nInputs .. $].dup;
    outNorm.normalize(rawOut);
    for(int j = 0; j < d.nTargets; ++j){
      assert(rawOut[j] == d.targets[i][j]);
    }
  }
  
  // Test unnormalizing Data
  for( int i = 0; i < d.nPoints; ++i){
    double[] rawIn = d.inputs[i].dup;
    inNorm.unnormalize(rawIn);
    for(size_t j = 0; j < d.nInputs; ++j){
      assert(rawIn[j] == testData[i][j]);
    }
    double[] rawOut = d.targets[i].dup;
    outNorm.unnormalize(rawOut);
    for(int j = 0; j < d.nTargets; ++j){
      assert(rawOut[j] == testData[i][j + d.nInputs]);
    }
  }
}

/**
 * Save a normalization to a file so it can be loaded back in later.
 * Useful for associating with a trained network, send the normalizations
 * along with the network file, otherwise the network is useless.
 * 
 * Params: 
 * norm = the normalization to save
 * path = path to the file to save.
 * 
 * TODO - When phobos library settles down, save these as XML instead.
 */
void SaveNormalization( Normalization norm, const string path){
  /*
   * File format:
   * Normalization
   * nVals = val                      // length of arrays shift and scale
   * shift = val,val,val,val,val,...
   * scale = val,val,val,val,val,...
   */

  assert(norm.scale.length == norm.shift.length);
  
  // Open the file
  Stream fl = new BufferedFile(path, FileMode.OutNew);
  scope(exit) fl.close();
  
  // Put a header to identify it as NormalizedData
  fl.writefln("Normalization");

  // The number of elements in the arrays
  ulong nVals = norm.scale.length;
  
  // Save the size
  fl.writefln("nVals = %d",nVals);

  // Save the arrays shift and scale
  fl.writef("shift = ");
  for(int i = 0; i < nVals - 1; ++i)
    fl.writef("%.*f,", double.dig, norm.shift[i]);
  fl.writef("%.*f\n", double.dig, norm.shift[$ - 1]);

  fl.writef("scale = ");
  for(int i = 0; i < nVals - 1; ++i)
    fl.writef("%.*f,", double.dig, norm.scale[i]);
  fl.writef("%.*f\n", double.dig, norm.scale[$ - 1]);
}

/**
 * Load a normalization from the file system.
 * 
 * Parms: 
 * fileName = path to the file to be loaded.
 * 
 * TODO - When phobos library settles down, use XML instead.
 */
Normalization LoadNormalization(const string fileName){
  // See file description in SaveNormalization

  // Open the file, read in the contents
  Stream fl = new BufferedFile(fileName, FileMode.In);
  string text = fl.toString();
  fl.close();
  
  // Split into lines
  string[] lines = split(text, regex("\n"));

  // clean up the lines
  foreach(ref line; lines) line = strip(line);
  
  // Parse number of values
  enforce(lines[0] == "Normalization");
  size_t nVals = to!size_t(lines[1][8 .. $]);

  // Set shift and scale
  double[] tmpShift = new double[](nVals);
  double[] tmpScale = new double[](nVals);

  // Parse shift
  string[] tokens = split(lines[2][8 .. $],regex(","));
  for(int i = 0; i < nVals; ++i) tmpShift[i] = to!double(strip(tokens[i]));

  // Parse scale
  tokens = split(lines[3][8 .. $],regex(","));
  for(int i = 0; i < nVals; ++i) tmpScale[i] = to!double(strip(tokens[i]));

  return Normalization(tmpShift, tmpScale);
}

unittest{
  mixin(announceTest("SaveNormalization and LoadNormalization Test."));

  Data d = LoadDataFromArray(testData, 5, 2, flags);
  
  auto inNorm = d.getNormalization!(DataType.INPUT_DATA);
  auto outNorm = d.getNormalization!(DataType.INPUT_DATA);
  
  SaveNormalization(inNorm, "inNorm.csv");
  scope(exit) std.file.remove("inNorm.csv");
  SaveNormalization(outNorm, "outNorm.csv");
  scope(exit) std.file.remove("outNorm.csv");
  
  auto loadedInNorm = LoadNormalization("inNorm.csv");
  auto loadedOutNorm = LoadNormalization("outNorm.csv");
  
  assert(approxEqual(inNorm.shift,loadedInNorm.shift));
  assert(approxEqual(inNorm.scale,loadedInNorm.scale));
  
}

