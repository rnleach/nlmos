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

version(unittest){

  // Set up some imports and utility functions for all unit tests.
  
  import std.stdio;
  import std.file;

  // Generate string for mixin that announces this test.
  string announceTest(in string msg){
    return "
    write(format(\"Testing %s - %5d: %s...\",__FILE__,__LINE__,\"" ~ msg ~"\"));
    scope(exit)writeln(\"done.\");";
  }  
}

/*------------------------------------------------------------------------------
 *                             DataPoint struct
 *----------------------------------------------------------------------------*/
/**
 * This struct is the most basic form of a data point, and is what
 * most methods in this framework know how to work with.
 * 
 * Authors: Ryan Leach
 * Date: January 21, 2015
 * See_Also: Data
 * 
 * Params:
 * numInputs  = The number of inputs stored in each DataPoint.
 * numTargets = The number of targets stored in each DataPoint.
 *
 */
 struct DataPoint(size_t numInputs, size_t numTargets){

  enum numVals = numInputs + numTargets;
  
  private double[numVals] data;

  /**
   * Params:
   * inpts = array of input values.
   * trgts = array of target values.
   */
  this(in double[] inpts, in double[] trgts){
    // Checks - debug releases only.
    assert(inpts.length == numInputs, "Length mismatch on DataPoint inpts.");
    assert(trgts.length == numTargets,"Length mismatch on DataPoint trgts.");

    // Since the values are stored in static arrays, must copy them elementwise.
    data[0 .. numInputs] = inpts[];
    data[numInputs .. numVals] = trgts[];

  }

  /**
   * Params:
   * vals = input and target values in single array with targets at the end of 
   *        of the array.
   */
  this(in double[] vals){
    // Checks - debug versions only
    assert(vals.length == numVals, "Length mismatch on DataPoint vals.");

    data[] = vals[];

  }

  /**
   * Params:
   * strRep = String representation of a DataPoint as produced by the stringRep
   *          method below.
   */
  this(in string strRep){

    // Regular expression for splitting the string on commas.
    auto sepRegEx = ctRegex!r",";

    // Split the string on commas
    string[] tokens = split(strRep, sepRegEx);

    // Check to make sure we have enough tokens.
    assert(tokens.length == numVals, "Length mismatch on DataPoint strRep.");

    // Copy in
    foreach(i; 0 .. numVals) 
      this.data[i] = to!double(tokens[i]);
  }

  // TODO add constructor that takes shift and scale already and applies it
  //      to the given data. Also requires a new normalize method that
  //      doesn't calculate the normalization parameters.

  /**
   * Returns: A string representation of the DataPoint.
   */
  @property string stringRep(){
    string toRet = "";

    foreach(val; this.data) 
      toRet ~= format("%.*f,", double.dig, val);
    
    return toRet[0 .. ($ - 1)]; // -1 to trim final comma
  }

  /**
   * Returns: A slice of just the inputs.
   */
  @property double[] inputs(){return this.data[0 .. numInputs];}

  /**
   * Returns: A slice of just the targets.
   */
  @property double[] targets(){return this.data[numInputs .. $];}

 }
 /*=============================================================================
  *                   Unit tests for DataPoint
  *===========================================================================*/
version(unittest){
  // Some values to keep around for testing DataPoint objects.
  double[5] inpts = [1.0, 2.0, 3.0, 4.0, 5.0];
  double[2] trgts = [6.0, 7.0];
  double[7] vals  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
}
unittest{
  mixin(announceTest("DataPoint this(double[], double[])"));

  DataPoint!(5,2) dp = DataPoint!(5,2)(inpts, trgts);
  assert(dp.data == vals);

  // DataPoint objects with no targets are possible too.
  DataPoint!(5,0) dp2 = DataPoint!(5,0)(inpts,[]);
  assert(dp2.data == inpts);
}
unittest{
  mixin(announceTest("DataPoint this(double[])"));

  DataPoint!(5,2) dp = DataPoint!(5,2)(vals);
  assert(dp.data == vals);

  // DataPoint objects with no targets are possible too.
  DataPoint!(5,0) dp2 = DataPoint!(5,0)(inpts);
  assert(dp2.data == inpts);
}
unittest{
  mixin(announceTest("DataPoint inputs and targets properties."));

  DataPoint!(5,2) dp = DataPoint!(5,2)(inpts, trgts);
  assert(dp.inputs == inpts);
  assert(dp.targets == trgts);

  // DataPoint objects with no targets are possible too.
  DataPoint!(5,0) dp2 = DataPoint!(5,0)(inpts,[]);
  assert(dp2.inputs == inpts);
  assert(dp2.targets == []);
}
unittest{
  mixin(announceTest("DataPoint stringRep"));

  DataPoint!(5,2) dp = DataPoint!(5,2)(vals);

  assert(dp.stringRep == "1.000000000000000,2.000000000000000," ~ 
                         "3.000000000000000,4.000000000000000," ~ 
                         "5.000000000000000,6.000000000000000," ~
                         "7.000000000000000");
}
unittest{
  mixin(announceTest("DataPoint this(string)"));

  DataPoint!(5,2) dp = DataPoint!(5,2)(vals);

  DataPoint!(5,2) dp2 = DataPoint!(5,2)(dp.stringRep);

  assert(dp == dp2);
}

/*******************************************************************************
 * A collection of DataPoint objects with some other added functionality 
 * related to creating various Ranges and Normalizations.
 *
 * Params:
 * numInputs  = The number of inputs stored in each DataPoint.
 * numTargets = The number of targets stored in each DataPoint.
 * 
 * Authors: Ryan Leach
 * Date: January 21, 2015
 * See_Also: DataPoint
 * 
*******************************************************************************/
class Data(size_t numInputs, size_t numTargets){

  // Compile time constant for convenience.
  enum numVals = numInputs + numTargets;

  // Shorthand for my datapoints
  alias DataPoint!(numInputs, numTargets) DP;

  private DP[] list;
  private size_t numPoints;
  private bool[] dataFilter;
  private double[numVals] shift;
  private double[numVals] scale;

  /**
   * Immutable constructor, only immutable instances of data should be used.
   * 
   * This constructor assumes the data is not normalized, and normalizes it.
   *
   * Params: 
   * data    = A 2-d array with rows representing a data point and columns, e.g. 
   *           1000 rows by 5 columns is 1000 5-dimensional data points. If
   *           there are 3 inputs and 2 targets, then the targets are the last
   *           two values in each point.
   * filter  = Must be size inputs + targets. True values indicate that the 
   *           corresponding column is a binary input (this is not checked) and 
   *           has value 0 or 1. Thus it should not be normalized.
   *
   * See_Also: Normalizations
   */
  private immutable this(const double[][] data, const bool[] filter){
    // Check lengths
    enforce(data.length > 1, 
      "Initialization Error, no points in supplied array.");
    
    enforce(numVals  == data[0].length,
      "Initialization Error, sizes do not match.");
    
    enforce(filter.length == data[0].length,
      "Initialization Error, filters do not match data.");
  
    this.numPoints = data.length;
    this.dataFilter = filter.idup;

    // Set up the local data storage and copy values
    //
    // list has scope only in this constructor, so when it exits there is no 
    // other reference to the newly copied data, except the immutable ones!
    DP[] temp = new DP[](numPoints);
    
    for(size_t i = 0; i < numPoints; ++i)
      temp[i] = DP(data[i]);
    
    // Normalize list in place while calculating shifts and scales
    double[numVals] shift_tmp;
    double[numVals] scale_tmp;
    normalize(temp, shift_tmp, scale_tmp, filter);

    // Cast temp to the immutable data, temp never escapes constructor as 
    // mutable data.
    this.list = cast(immutable) temp;

    this.shift = cast(immutable)shift_tmp;
    this.scale = cast(immutable)scale_tmp;

  }

  /**
   * Immutable constructor, only immutable instances of data should be used. 
   *
   * This constructor assumes the data IS already normalized, and does not 
   * normalize it again. Intended to be used when loading normalizded data from
   * a file.
   * 
   * Params: 
   * data    = A 2-d array with rows representing a data point and columns,
   *                e.g. 1000 rows by 5 columns is 1000 5-dimensional data 
   *                points.
   * filter  = Must be size inputs + targets. True values indicate that the 
   *           corresponding column is a binary input (this is not checked) and 
   *           has value 0 or 1. Thus it should not be normalized. This is used
   *           for creating Normalizations, so it is still needed.
   * shift   = The shift used in the normalization.
   * scale   = The scale used in the normalization.
   *
   * See_Also: Normalizations.
   */
  private immutable this(const double[][] data, 
                         const bool[] filter,
                         const double[] shift,
                         const double[] scale){
    
    // Check lengths
    enforce(data.length > 1, 
      "Initialization Error, no points in supplied array.");
    
    enforce(filter.length == numVals, 
      "Initialization Error, filters do not match data.");
    
    enforce(shift.length == numVals, 
      "Initialization Error, shifts do not match data.");
    
    enforce(scale.length == numVals, 
      "Initialization Error, scales do not match data.");
    
    // Assign values
    this.numPoints = data.length;
    this.dataFilter = filter.idup;
    this.scale = scale.idup;
    this.shift = shift.idup;
    
    // tmp has scope only in this constructor, so when it exits there
    // is no other reference to the newly copied data, except the immutable
    // one!
    DP[] tmp = new DP[](numPoints);
    
    // Copy data by value into tmp arrays
    foreach(size_t i, const double[] d; data) 
      tmp[i] = DP(d);
    this.list = cast(immutable) tmp;
  }

  /**
   * Normalizes the array dt in place, returning the shift and scale of the
   * normalization in the so-named arrays. Filters are used to mark binary 
   * inputs and automatically set their shift to 0 and scale to 1.
   *
   * TODO parallelize this section to speed it up if possible.
   */
  private final void normalize(DP[] dt, 
                         ref double[numVals] shift, 
                         ref double[numVals] scale,
                         in bool[] filters) immutable {

    double[numVals] sum;
    double[numVals] sumsq;
    
    // Initialize
    shift[] = 0.0;
    scale[] = 1.0;
    sum[] = 0.0;
    sumsq[] = 0.0;
    
    // Calculate the sum and sumsq
    foreach(d; dt){
      for(size_t i = 0; i < numVals; ++i){
        if(!filters[i]){
          sum[i] += d.data[i];
          sumsq[i] += d.data[i] * d.data[i];
        }
      }
    }
    
    // Calculate the mean (shift) and standard deviation (scale)
    size_t nPoints = dt.length;
    for(size_t i = 0; i < numVals; ++i){
      if(!filters[i]){
        shift[i] = sum[i] / nPoints;
        scale[i] = sqrt((sumsq[i] / nPoints - shift[i] * shift[i]) * 
          nPoints / (nPoints - 1));
      }
    }
    
    // Now use these to normalize the data
    foreach(ref d; dt){
      for(size_t j = 0; j < numVals; ++j)
        d.data[j] = (d.data[j] - shift[j]) / scale[j];
    }
    
    // All done, now return.
  }

  /**
   * Get information about the sizes of data associated with this Data object.
   */
  @property final size_t nPoints() const {return this.numPoints;}
  /**
   * ditto
   */
  @property final size_t nInputs() const {return numInputs;}
  /**
   * ditto
   */
  @property final size_t nTargets() const {return numTargets;}

  /**
   * Returns: a range that iterates over the points in this collection in
   *          the same order everytime.
   */
  @property final DataRange!(numInputs, numTargets) simpleRange() const{
    return DataRange!(numInputs, numTargets)(this);
  }

  /**
   * Returns: The DataPoint object at the given position in this collection.
   */
  public final DP getPoint(size_t index) const {return this.list[index];}

}
/*==============================================================================
 *                     Unit tests for data class
 *============================================================================*/
version(unittest){
  
  // Test Data
  double[][] testData = 
    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
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

  double scalePar = 0.302765035409750;
  double[] shiftPar = [1.45, 2.45, 3.45, 4.45, 5.45, 6.45, 7.45];
  
  // None of these values are binary, so all flags are false
  bool[] flags = [false, false, false, false, false, false, false];
  
}
unittest{
  mixin(announceTest("Data this(double[][], bool[])"));

  // Short-hand for dealing with immutable data
  alias immutable(Data!(5,2)) iData;
  
  iData d = new iData(testData, flags);
  
  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(approxEqual(sc, scalePar));

  foreach(i; 0 .. d.numVals) 
    assert(approxEqual(shiftPar[i], d.shift[i]));

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i){
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(approxEqual(d.list[i].data[j], normalizedRowValues[i]));
  }
}
unittest{
  mixin(announceTest("Data this(double[][], bool[], double[], double[])"));

  // Short-hand for dealing with immutable data
  alias immutable(Data!(5,2)) iData;

  double[][] normTestData = new double[][](10, 7);
  foreach(i; 0 .. 10){
      normTestData[i][] = normalizedRowValues[i];
  }
  double[7] scaleArr = scalePar;

  iData d = new iData(normTestData, flags, shiftPar, scaleArr);

  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(approxEqual(sc, scalePar));

  foreach(i; 0 .. d.numVals) 
    assert(approxEqual(shiftPar[i], d.shift[i]));

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i){
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(approxEqual(d.list[i].data[j], normalizedRowValues[i]));
  }
}
unittest{
  mixin(announceTest("Data nPoints, nInputs, nTargets properties."));
  
  // Short-hand for dealing with immutable data
  alias immutable(Data!(5,2)) iData;
  
  iData d = new iData(testData, flags);
  
  assert(d.nPoints == 10);
  assert(d.nInputs == 5);
  assert(d.nTargets == 2);

}
unittest{
  mixin(announceTest("Data getPoint(size_t)."));
  
  // Short-hand for dealing with immutable data
  alias immutable(Data!(5,2)) iData;
  alias immutable(DataPoint!(5,2)) iDataPoint;
  
  iData d = new iData(testData, flags);

  foreach(i; 0 .. d.nPoints){
    iDataPoint dp = d.getPoint(i);
    foreach(j; 0 .. dp.data.length){
      assert(approxEqual(dp.data[j], normalizedRowValues[i]));
    }
  }
}

/*==============================================================================
 *                     Helper Functions for Data
 *============================================================================*/
/**
 * Get a Data object from the provided array.
 *
 * Params: 
 * numIn   = The number of values in a sample that are inputs. These 
 *           are assumed to be the first numIn values in each row.
 * numTarg = The number of values in a sample that are targets. These
 *           are assumed to be the last numTarg values in each row.
 *           numIn + numTarg must equal the length of each row.
 * d       = The array to manage as Data. Each row is considered point
 *           or a sample.
 * filters = Sometimes inputs/targets are binary, and  you don't want 
 *           them to be normalized. For each input/target column that 
 *           is binary the corresponding value in the filters array
 *           is true. The length of the filters array must be numIn + numTarg. 
 */
auto LoadDataFromArray(size_t numInputs, size_t numTargets)
                      (const double[][] d, const bool[] filters){
  return new immutable(Data!(numInputs, numTargets))(d, filters);
}
unittest{
  mixin(announceTest("LoadDataFromArray(double[][], bool[])"));

  // Short-hand for dealing with immutable data
  alias immutable(Data!(5,2)) iData;
  
  iData d = LoadDataFromArray!(5,2)(testData, flags);
  
  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(approxEqual(sc, scalePar));

  foreach(i; 0 .. d.numVals) 
    assert(approxEqual(shiftPar[i], d.shift[i]));

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i){
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(approxEqual(d.list[i].data[j], normalizedRowValues[i]));
  }
}

/**
 * Load data from a file and create a Data object.
 *
 * Params:  
 * numInputs  = The number of values in a sample that are inputs. These 
 *              are assumed to be the first numIn values in each row.
 * numTargets = The number of values in a sample that are targets. These
 *              are assumed to be the last numTarg values in each row.
 *              numInputs + numTargets must equal the length of each row.
 * filename   = Path to the data to load
 * filters    = Sometimes inputs/targets are binary, and  you don't want 
 *              them to be normalized. For each input/target column that 
 *              is binary the corresponding value in the filters array is true.
 *              The length of the filters array must be numInputs + numTargets.
 */
auto LoadDataFromCSVFile(size_t numInputs, size_t numTargets)
                        (const string filename, bool[] filters){

  // Compile time value, convenient to keep around.
  enum numVals = numInputs + numTargets;
  
  // Open the file
  Stream f = new BufferedFile(filename);
  scope(exit) f.close();

  // Read the file line by line
  auto app = appender!(double[][])();
  auto sepRegEx = ctRegex!r",";
  foreach(size_t lm, char[] line; f){
    
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
  return new immutable(Data!(numInputs, numTargets))(app.data, filters);
}
unittest{
  mixin(announceTest("LoadDataFromCSVFile(string, bool[])"));

  // Not a good unittest, it relies on an external file.
  bool[] filters2 = [false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false];

  auto d = LoadDataFromCSVFile!(21,1)("MissoulaTempAllData.csv", filters2);

  // If no exceptions are thrown, this unit test passes for now. More tests
  // will be done in later unit tests.
}

// TODO saveProcessedData
// TODO loadProcessedData
// TODO isProcessedData
// TODO split - given a Data object, split it into two without re-normalizing

/*==============================================================================
 *                                   DataRange
 *============================================================================*/
/**
 * InputRange for iterating Data objects.
 */
public struct DataRange(size_t numInputs, size_t numTargets){

  alias const(Data!(numInputs, numTargets)) iData;

  private immutable(size_t) length;
  private size_t next;
  private iData data;
  
  /**
   * Params: 
   * d = The Data object you wish to iterate over.
   */
  this(iData d){
    this.length = d.nPoints;
    this.next = 0;
    this.data = d;
  }
  
  // Properties/methods to make this an InputRange
  @property bool empty(){return next == length;}

  @property auto front(){return this.data.getPoint(next);}

   void popFront(){++next;}
}
static assert(isInputRange!(DataRange!(5,2)));

unittest{
  mixin(announceTest("DataRange"));

  alias immutable(Data!(5, 2)) iData;

  iData d = LoadDataFromArray!(5,2)(testData, flags);

  auto r = DataRange!(5,2)(d);
  size_t i = 0;
  foreach(t; r) assert(t == d.getPoint(i++));
}
unittest{
  mixin(announceTest("Data simpleRange property"));

  alias immutable(Data!(5, 2)) iData;

  iData d = LoadDataFromArray!(5,2)(testData, flags);

  auto r = d.simpleRange;
  size_t i = 0;
  foreach(t; r) assert(t == d.getPoint(i++));

}

// TODO Random Data Range - data range that randomizes iteration.

/+ Old section 


class mData
{
	
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
	 
}

/*-----------------------------------------------------------------------------
 *                           Data Helper Functions
 *---------------------------------------------------------------------------*/

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

+/