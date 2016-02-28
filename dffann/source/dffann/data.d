/**
 * Utilities for loading, saving, and passing data around.
 *
 * Author: Ryan Leach
 * 
 */
module dffann.data;

import std.algorithm;
import std.array;
import std.container;
import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.random;
import std.range;
import std.regex;
import std.stdio;
import std.string;

version(unittest)
{
  import dffann.testutilities.testdata;
}

/*------------------------------------------------------------------------------
 *                             DataPoint struct
 *----------------------------------------------------------------------------*/
/**
 * This struct is the most basic form of a data point.
 * 
 * Authors: Ryan Leach
 * Date: February 27, 2016
 * Version: 2.0
 * See_Also: Data
 *
 * History:
 *    V1.0 Initial implementation.
 *    V2.0 No longer a template, instead of holding data, holds a view into a
 *         Data object via slices.
 *
 */
public struct DataPoint
{

  public double[] inputs;
  public double[] targets;

  /**
   * Params:
   * inpts = array of input values.
   * trgts = array of target values.
   */
  this(double[] inpts, double[] trgts = [])
  {
    inputs = inpts;
    targets = trgts;
  }

  this(immutable double[] inpts, immutable double[] trgts) immutable
  {
    inputs = inpts;
    targets = trgts;
  }

  /**
   * Returns: A string representation of the DataPoint.
   */
  @property string stringRep() const
  {
    string toRet = "";

    foreach(val; this.inputs) 
      toRet ~= format("%.*f,", double.dig, val);
    foreach(val; this.targets)
      toRet ~= format("%.*f,", double.dig, val);
    
    return toRet[0 .. ($ - 1)]; // -1 to trim final comma
  }
}

version(unittest)
{
  // Some values to keep around for testing DataPoint objects.
  double[7] vals  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
  double[] inpts = vals[0 .. 5];
  double[] trgts = vals[5 .. $];
}
///
unittest
{
  const DataPoint dp = DataPoint(inpts, trgts);
  assert( dp.inputs == vals[0 .. 5] );

  // DataPoint objects with no targets are possible too.
  const DataPoint dp2 = DataPoint(vals);
  assert( dp2.inputs == vals );
}

///
unittest
{
  const DataPoint dp = DataPoint(vals);

  assert(dp.stringRep == "1.000000000000000,2.000000000000000," ~ 
                         "3.000000000000000,4.000000000000000," ~ 
                         "5.000000000000000,6.000000000000000," ~
                         "7.000000000000000");
}

/*******************************************************************************
 * A collection of DataPoint objects with some other added functionality 
 * related to creating various Ranges and Normalizations.
 * 
 * Authors: Ryan Leach
 * Date: January 21, 2015
 * See_Also: DataPoint
 * Version: 2.0
 *
 * History:
 *    V1.0 Initial implementation.
 *    V2.0 No longer a template. Moved normalization out into it's own class.
 *
 * 
*******************************************************************************/
public class Data
{

  private double[] data_;
  private const uint numInputs;
  private const uint numTargets;
  private const uint numVals;

  /**
   * Factory method to create immutable versions of data.
   */
  public static immutable(Data) createImmutableData(uint nInputs,
    uint nTgts, in double[][] data)
  {
    return cast(immutable) new Data(nInputs, nTgts, data);
  }

  /**
   * Basic constructor
   *
   * Params: 
   * nInputs = The number of values in a point that are inputs. This class 
   *           assumes the inputs are first, followed by the targets in an
   *           array.
   * nTgts   = The number of values in a point that are targets. This class
   *           assumes those are the last points in an array.
   * data    = A 2-d array with rows representing a data point and columns, e.g. 
   *           1000 rows by 5 columns is 1000 5-dimensional data points. If
   *           there are 3 inputs and 2 targets, then the targets are the last
   *           two values in each point.
   *
   */
  public this(uint nInputs, uint nTgts, in double[][] data)
  {
    this.numInputs  = nInputs;
    this.numTargets = nTgts;
    this.numVals    = nInputs + nTgts;

    // Check lengths
    enforce(data.length > 1, 
      "Initialization Error, no points in supplied array.");
    
    enforce(this.numVals  <= data[0].length,
      "Initialization Error, sizes do not match.");
    
    // Set up the local data storage and copy values
    //
    // list has scope only in this constructor, so when it exits there is no 
    // other reference to the newly copied data, except the immutable ones!
    this.data_ = new double[](data.length);
    
    for(size_t i = 0; i < data.length; ++i)
    {
      size_t start = i * this.numVals;
      assert( data[i].length == this.numVals );
      this.data_[start .. (start + this.numVals)] = data[i][];
    }
  }

  /**
   * Get information about the sizes of data associated with this Data object.
   */
  @property final size_t nPoints() const 
  {
    return this.data_.length / this.numVals;
  }

  /// ditto
  @property final size_t nInputs() const {return this.numInputs;}
  
  /// ditto
  @property final size_t nTargets() const {return this.numTargets;}


  /**
   * Returns: a range that iterates over the points in this collection.
   */
  @property final auto simpleRange() const
  {
    return DataRange!(typeof(this))(this);
  }

  /**
   * Returns: a range that iterates over the points in this collection in
   *          the same order everytime.
   */
  @property final auto infiniteRange() const
  {
    return InfiniteDataRange!(typeof(this))(this);
  }

  /**
   * Returns: a range that iterates over the points in this collection at
   *          random. It is an infinite range.
   */
  @property final auto randomRange() const
  {
    return RandomDataRange!(typeof(this))(this);
  }

  /**
   * Returns: The DataPoint object at the given position in this collection.
   */
  public final inout (DataPoint) opIndex(size_t index) inout 
  {
    const size_t start = index * numVals;
    const size_t end = start + numVals;
    const size_t brk = start + numInputs;
    return DataPoint(data_[start .. brk], data_[brk .. end]);
  }

  public final size_t opDollar(size_t pos)() const
  {
    return this.data_.length;
  }

  /+

  /**
   * Load data from a file and create a Data object.
   *
   * Params:  
   * filename   = Path to the data to load
   * filters    = Sometimes inputs/targets are binary, and  you don't want 
   *              them to be normalized. For each input/target column that 
   *              is binary the corresponding value in the filters array is true.
   *              The length of the filters array must be numInputs + numTargets.
   */
  public static final Data loadDataFromCSVFile
                                      (const string filename, bool[] filters)
  {

    // Compile time value, convenient to keep around.
    enum numVals = numInputs + numTargets;
    
    // Open the file
    File f = File(filename, "r");

    // Read the file line by line
    auto app = appender!(double[][])();
    auto sepRegEx = ctRegex!r",";
    foreach(char[] line; f.byLine)
    {
      
      // Split the line on commas
      char[][] tokens = split(line,sepRegEx);
      
      // Ensure we have enough tokens to do the job, and that the line is
      // numeric. This should skip any kind of 'comment' line or column headers
      // that start with non-numeric strings.
      if(tokens.length != numVals || !isNumeric(tokens[0])) continue;
      
      // Parse the doubles from the strings
      double[] lineValues = new double[](numVals);
      for(size_t i = 0; i < numVals; ++i)
        lineValues[i] = to!double(tokens[i]);
      
      // Add them to my array
      app.put(lineValues);
    }
    
    // Return the new Data instance
    return new Data(app.data, filters);
  }

  /** 
   * Save normalized data in a pre-determined format so that it 
   * can be loaded quickly without the need to re-calculate the normalization.
   * 
   * Params: 
   * pData    = The data object to save.
   * filename = The path to save the data.
   */
  public static final void saveProcessedData
             (const Data pData, const string filename)
  {
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
    File fl = File(filename, "w");

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
    string insertWriteArray(string vname, string format, string precis = "")
    {
      return "
        fl.writef(\"" ~ vname ~ " = \");
        for(int i = 0; i < pData." ~ vname ~ ".length - 1; ++i)
          fl.writef(\"" ~ format ~ ",\"" ~ precis ~ ", pData." ~ vname ~ "[i]);
        fl.writef(\"" ~ format ~ "\n\"" ~ precis ~ ", pData." ~ vname ~ "[$ - 1]);
      ";
    }

    // Save the filters and normalizations 
    mixin(insertWriteArray("dataFilter","%s"));
    mixin(insertWriteArray("shift","%.*f",",double.dig"));
    mixin(insertWriteArray("scale","%.*f",",double.dig"));

    // Now write the data points
    fl.writefln("data = ");
    foreach(dp; pData.list)
      fl.writefln(dp.stringRep);
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
  public static final Data loadProcessedData(const string filename)
  {

    // See comments in SaveProcessedData for file and header formats.

    // Open the file, read in the contents
    string text = readText(filename);

    // Split into lines as header and data sections
    string[] lines = split(text, regex("\n"));
    string[] header = lines[0 .. 8];
    string[] dataLines = lines[8 .. $];

    // clean up the header lines
    foreach(ref line; header) line = strip(line);

    // Parse some variables out of the header section
    size_t nmPoints = to!size_t(header[1][10 .. $]);
    size_t nmInputs = to!size_t(header[2][10 .. $]);
    size_t nmTargets = to!size_t(header[3][11 .. $]);
    size_t totalVals = numInputs + numTargets;

    // Check that this file matches the kind of Data object we are loading.
    enforce(header[0] == "NormalizedData");
    enforce(nmInputs == numInputs, "File number of inputs does not match.");
    enforce(nmTargets == numTargets, "File number of targets does not match.");

    // set up some variables, define nested function for mixin to
    // parse arrays.
    string[] tokens;
    // Parms: vname  - e.g. "dataFilter"
    //        elType - e.g. "bool", "double"
    //        row    - e.g. "4", row number of header array to parse.
    string insertParseArray(string vname, string elType, string row)
    {
      string startCol = to!string(vname.length + 2);

      return "tokens = split(header["~row~"]["~startCol~" .. $],regex(\",\"));
        "~vname~" = new "~elType~"[](numVals);
        for(int i = 0; i < numVals; ++i) 
          "~vname~"[i] = to!"~elType~"(strip(tokens[i]));";
    }

    bool[] lDataFilter;
    mixin(insertParseArray("lDataFilter","bool","4"));

    double[] lShift;
    mixin(insertParseArray("lShift","double","5"));

    double[] lScale;
    mixin(insertParseArray("lScale","double","6"));

    // Now parse each row
    enforce(dataLines.length >= nmPoints,
      "Malformed data file, not enough input points.");

    double[][] tmp = new double[][](nmPoints,numVals);
    
    foreach(i; 0 ..nmPoints)
    {

      tokens = split(dataLines[i], regex(","));

      enforce(tokens.length >= totalVals,
        "Malformed data file, not enought points on line.");

      foreach(j; 0 .. numVals)
        tmp[i][j] = to!double(strip(tokens[j]));

    }

    return new Data(tmp,  lDataFilter,  lShift, lScale);
  }

  /**
   * Check if the supplied path is to a data file that has been pre-processed or
   * not.
   * 
   * Params: 
   * filename = The path to the file to check.
   * 
   * Returns: true if the file is laid out as expected for something that has
   * been saved with SaveProcessedData.
   * 
   */
  public static final bool isProcessedDataFile(const string filename)
  {

    // See comments in SaveProcessedData for file and header formats.

    // Open the file, read in the contents
    try{

      string text = readText(filename);

      // Split into lines as header and data sections
      string[] lines = split(text, regex("\n"));
      string[] header = lines[0 .. 8];
      const string[] dataLines = lines[8 .. $];

      // clean up the header lines
      foreach(ref line; header) line = strip(line);

      // Parse some variables out of the header section
      const size_t nmPoints = to!size_t(header[1][10 .. $]);
      const size_t nmInputs = to!size_t(header[2][10 .. $]);
      const size_t nmTargets = to!size_t(header[3][11 .. $]);

      // Check that this file matches the kind of Data object we are loading.
      if(header[0] != "NormalizedData") return false;
      if(nmInputs != numInputs) return false;
      if(nmTargets != numTargets) return false;
      if(dataLines.length < nmPoints) return false;

    }
    catch(Exception e){
      return false;
    }

    return true;
  }
  +/
}

/*==============================================================================
 *                     Unit tests for data class
 *============================================================================*/
version(unittest)
{
  
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

/+
unittest
{
  // Short-hand for dealing with immutable data
  alias iData = immutable(Data!(5,2));
  
  iData d = new iData(testData, flags);
  
  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(approxEqual(sc, scalePar),format("%f != %f", sc, scalePar));

  foreach(i; 0 .. d.numVals) 
    assert(approxEqual(shiftPar[i], d.shift[i]));

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i)
  {
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(approxEqual(d.list[i].data[j], normalizedRowValues[i]));
  }
}
unittest
{
  // Short-hand for dealing with immutable data
  alias iData = immutable(Data!(5,2));
  
  iData d = Data!(5,2).createImmutableData(testData, flags, false);
  
  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(sc == 1.0);

  foreach(sh; d.shift) 
    assert(sh == 0.0);

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i)
  {
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(d.list[i].data[j] == testData[i][j]);
  }
}
unittest
{
  // Short-hand for dealing with immutable data
  alias iData = immutable(Data!(5,2));

  double[][] normTestData = new double[][](10, 7);
  foreach(i; 0 .. 10){
      normTestData[i][] = normalizedRowValues[i];
  }
  const double[7] scaleArr = scalePar;

  iData d = new iData(normTestData, flags, shiftPar, scaleArr);

  // Check the number of points
  assert(d.numPoints == 10);

  // Check the shift and scale - to be used later to create Normalizations.
  foreach(sc; d.scale) 
    assert(approxEqual(sc, scalePar));

  foreach(i; 0 .. d.numVals) 
    assert(approxEqual(shiftPar[i], d.shift[i]));

  // Test normalization of data points
  for(size_t i = 0; i < d.numPoints; ++i)
  {
    for(size_t j = 0; j < d.numVals; ++j) 
      assert(approxEqual(d.list[i].data[j], normalizedRowValues[i]));
  }
}
unittest
{
  // Short-hand for dealing with immutable data
  alias iData = immutable(Data!(5,2));
  
  const iData d = Data!(5,2).createImmutableData(testData, flags);
  
  assert(d.nPoints == 10);
  assert(d.nInputs == 5);
  assert(d.nTargets == 2);
}
unittest
{
  // Short-hand for dealing with immutable data
  alias iData = immutable(Data!(5,2));
  alias iDataPoint = immutable(DataPoint!(5,2));
  
  iData d = Data!(5,2).createImmutableData(testData, flags);

  foreach(i; 0 .. d.nPoints)
  {
    iDataPoint dp = d.getPoint(i);
    foreach(j; 0 .. dp.data.length)
      assert(approxEqual(dp.data[j], normalizedRowValues[i]));
  }
}
unittest
{
  alias iData = immutable(Data!(21,1));

  bool[] filters2 = [false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false];

  makeRandomCSVFile(222, filters2,"TestData.csv");
  scope(exit) std.file.remove("TestData.csv");

  Data!(21,1).loadDataFromCSVFile("TestData.csv", filters2);

  // If no exceptions are thrown, this unit test passes for now. More tests
  // will be done in later unit tests.
}
unittest
{
  alias Data21 = Data!(21,1);

  enum string testFileName = "TestData.csv";
  enum string testFileNameForNormalizedData = "TestDataNrm.csv";

  // Not a good unittest, it relies on an external file.
  bool[] filters2 = [false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,
                     false];

  makeRandomCSVFile(222, filters2, testFileName);
  scope(exit) std.file.remove(testFileName);
  
  assert(exists(testFileName));
  assert(!Data21.isProcessedDataFile(testFileName));

  Data21 d2 = Data21.loadDataFromCSVFile(testFileName, filters2);
  
  Data21.saveProcessedData(d2, testFileNameForNormalizedData);
  scope(exit) std.file.remove(testFileNameForNormalizedData);
  assert(Data21.isProcessedDataFile(testFileNameForNormalizedData));
  
  auto d3 = Data21.loadProcessedData(testFileNameForNormalizedData);
  assert(d2.nPoints == d3.nPoints);
  assert(d2.nInputs == d3.nInputs);
  
  foreach(k; 0 .. d3.nPoints)
    assert(approxEqual(d2.getPoint(k).data[],d3.getPoint(k).data[]));
}
+/
/*==============================================================================
 *                                   DataRange
 *============================================================================*/

/**
 * ForwardRange for iterating over a Data object.
 */
public struct DataRange(DataType)
{

  private size_t start_;
  private size_t end_;
  private DataType dataObject_;

  /**
   * Params: 
   * d = The Data object you wish to iterate over.
   */
  this(DataType d)
  {
    this.start_ = 0;
    this.end_ = d.nPoints;
    this.dataObject_ = d;
  }
  
  /// Properties/methods to make this a RandomAccessRange.
  @property bool empty(){ return start_ >= end_; }

  /// ditto
  @property auto ref front(){ return this.dataObject_[start_]; }

  /// ditto
  void popFront(){ ++start_; }

  /// ditto
  @property auto ref back(){ return this.dataObject_[end_ - 1]; }

  /// ditto
  void popBack(){ --end_; }

  /// ditto
  // Since this is a struct, return copies all values! Easy.
  @property auto save(){ return this; }

  auto ref opIndex(size_t index)
  {
    return this.dataObject_[start_ + index];
  }

  /// ditto
  @property size_t length(){return end_ - start_;}

  /// ditto
  /*
  Never have been able to get this to work with the hasSlicing! test.
  */
  typeof(this) opSlice(size_t start, size_t end)
  {
    assert(start <= end, "Range Error, start must be less than end.");
    assert(start >= 0, "No negative indicies allowed!");
    assert( this.start_ + end < this.end_, "Out of range!" );

    // Create a copy
    auto temp = this.save;

    // Update the next
    temp.start_  = start_ + start;
    temp.end_   = start_ + end;

    assert(temp.start_ <= this.dataObject_.nPoints);
    assert(temp.end_   <= this.dataObject_.nPoints);

    return temp;
  }

  /// ditto
  @property size_t opDollar(){ return this.length; }
}
static assert( isInputRange!(DataRange!(immutable Data)) );
static assert( isForwardRange!(DataRange!(immutable Data)) );
static assert( isBidirectionalRange!(DataRange!(immutable Data)) );
static assert( isRandomAccessRange!(DataRange!(immutable Data)) );
static assert( hasLength!(DataRange!(immutable Data)) );

///
unittest
{
  // Create a data set to test on.
  auto d = Data.createImmutableData(5, 2, testData);

  // Create a range to test
  auto r = DataRange(d);

  // Check length
  assert(testData.length == r.length);

  // Test that the elements line up.
  size_t i = 0;
  foreach(t; r) assert( t == d[i++]);
  
  // TODO a lot more tests on this range!
}

/**
 * Infinite ForwardRange for iterating a data set in a random order.
 */
public struct RandomDataRange(DataType)
{
  private DataType dataObject_;
  private const size_t nPoints_;


  /**
   * Params: 
   * d = The Data object you wish to iterate over.
   */
  this(DataType d)
  {
    dataObject_ = d;
    nPoints_ = d.nPoints;
  }
  
  /// Properties/methods to make this a ForwardRange with slicing.
  public enum bool empty = false;

  /// ditto
  @property auto ref front()
  {
    return this.dataObject_[uniform(0, nPoints_)];
  }

  /// ditto
  void popFront(){}

  /// ditto
  // Since this is a struct, return copies all values! Easy.
  @property auto save() { return this; }
}
static assert( isForwardRange!(RandomDataRange!(immutable Data)) );
static assert( isInfinite!(RandomDataRange!(immutable Data)) );

/**
 * Infinite ForwardRange for iterating a data set over and over in the same 
 * order.
 */
public struct InfiniteDataRange(DataType)
{
  private size_t next_;
  private const size_t end_;
  private DataType dataObject_;

  /**
   * Params: 
   * d = The Data object you wish to iterate over.
   */
  this(DataType d)
  {
    this.next_ = 0;
    this.end_ = d.nPoints;
    this.dataObject_ = d;
  }
  
  /// Properties/methods to make this a RandomAccessRange with slicing.
  public enum bool empty = false; 

  /// ditto
  @property auto ref front(){ return this.dataObject_[next_]; }

  /// ditto
  void popFront()
  { 
    ++next_;
    if(next_ >= end_) next_ = 0;
  }

  /// ditto
  // Since this is a struct, return copies all values! Easy.
  @property auto save(){ return this; }
}
static assert( isForwardRange!(InfiniteDataRange!(immutable Data)) );
static assert( isInfinite!(InfiniteDataRange!(immutable Data)) );
/+
/*==============================================================================
 *                                Normalizations
 *==============================================================================
 * Will use a struct for this, but considered using closures. I foresee 
 * situations where this is called a lot, so a closure might be pretty 
 * inefficient considering a struct can be allocated on the stack, whereas a
 * closure would necessarily be allocated on the heap.
 */

/**
 * A normalization is used to force data to have certain statistical properties,
 * such as a standard deviation of 1 and a mean of 0. 
 * 
 * Sometimes data is binary, e.g. with values 0 or 1, or possibly -1 or 1, in 
 * which case you do not want to normalize it. Binary data may also be mixed 
 * with non-binary data. This is handled at the constructor level when loading
 * the data with the use of filters to determine if a column is binary.
 */
   /**
   * Normalizes the array dt in place, returning the shift and scale of the
   * normalization in the so-named arrays. Filters are used to mark binary 
   * inputs and automatically set their shift to 0 and scale to 1.
   */
  private final void normalize(DP[] dt, 
                         ref double[numVals] shift, 
                         ref double[numVals] scale,
                         in bool[] filters) const
  {

    /*==========================================================================
      Nested struct to hold results of summing over a batch
    ==========================================================================*/
    struct BatchResults {
      public double[numVals] batchSum;
      public double[numVals] batchSumSquares;

      public this(const double[numVals] sum, const double[numVals] sumSq)
      {
        this.batchSum = sum;
        this.batchSumSquares = sumSq;
      }
    }

    /*==========================================================================
      Nested function to calculate a batch of stats.
    ==========================================================================*/
    BatchResults sumChunk(DP[] chunk, in bool[] filters) pure
    {
      double[numVals] sm = 0.0;
      double[numVals] smSq = 0.0;

      // Calculate the sum and sumsq
      foreach(d; chunk){
        for(size_t i = 0; i < numVals; ++i)
        {
          if(!filters[i])
          {
            sm[i] += d.data[i];
            smSq[i] += d.data[i] * d.data[i];
          }
        }
      }

      return BatchResults(sm, smSq);
    }

    /*==========================================================================
      Now do the summations in parallel.
    ==========================================================================*/
    double[numVals] sum = 0.0;
    double[numVals] sumsq = 0.0;

    // How many threads to use?
    size_t numThreads = totalCPUs - 1;
    if(numThreads < 1) numThreads = 1;

    BatchResults[] reses = new BatchResults[numThreads];
    const size_t chunkSize = dt.length / numThreads;
    size_t[] starts = new size_t[numThreads];
    size_t[] ends = new size_t[numThreads];
    foreach(i; 0 .. numThreads)
    {
      if( i == 0)
      {
        starts[i] = 0;
      } 
      else
      {
        starts[i] = ends[i - 1];
      }
      ends[i] = starts[i] + chunkSize + (i < (dt.length % numThreads) ? 1 : 0);
    }

    foreach(i, ref res; parallel(reses))
    {
      res = sumChunk(dt[starts[i] .. ends[i]] , filters);
    }

    // Initialize
    shift[] = 0.0;
    scale[] = 1.0;
    sum[] = 0.0;
    sumsq[] = 0.0;

    foreach(i, res; reses)
    {
      sum[] += res.batchSum[];
      sumsq[] += res.batchSumSquares[];
    }
    
    // Calculate the mean (shift) and standard deviation (scale)
    size_t nPoints = dt.length;
    for(size_t i = 0; i < numVals; ++i)
    {
      if(!filters[i])
      {
        shift[i] = sum[i] / nPoints;
        scale[i] = sqrt((sumsq[i] / nPoints - shift[i] * shift[i]) * 
          nPoints / (nPoints - 1));
      }
    }
    
    // Now use these to normalize the data
    foreach(ref d; parallel(dt)){
      for(size_t j = 0; j < numVals; ++j)
        d.data[j] = (d.data[j] - shift[j]) / scale[j];
    }
    
    // All done, now return.
  }

struct Normalization
{

  private double[] shift;
  private double[] scale;
  
  /**
  * Params: 
  * shift = array of shifts to subtract from each point to be normalized.
  * scale = array of scales to divide each point by.
  */
  private this(const double[] shift, const double[] scale)
  {
    enforce(shift.length == scale.length, 
      "Shift and scale array lengths differ.");

    this.shift = shift.dup;
    this.scale = scale.dup;
  }

  /**
   * Given a normalized input/target set of values, unnormalize them.
   */
  void unnormalize(T)(ref T d) if(T.stringof.find("DataPoint"))
  {
    assert(d.data.length <= shift.length && d.data.length <= scale.length);

    d.data[] = d.data[] * scale[0 .. d.data.length] + shift[0 .. d.data.length];
  }

  /**
   * Given a normalized output (from a network) unnormalize the outputs. Assume
   * these are outputs, so unnormalize against the end of the shift and scale
   * arrays, since arrays are packed [inputs, targets]. So this if for 
   * unnormalizing the outputs of a network.
   */
  void unnormalize(double[] d)
  {
    d[] = d[] * scale[($ - d.length) .. $] + shift[($ - d.length) .. $];
  }
  
  /**
   * Given an unormalized input/output set of values, normalize them.
   */
  void normalize(T)(ref T d) if(T.stringof.find("DataPoint")) 
  {
    assert(d.data.length <= shift.length && d.data.length <= scale.length);

    d.data[] = 
            (d.data[] - shift[0 .. d.data.length]) / scale[0 .. d.data.length];
  }
}

///
unittest
{
  Data!(5, 2) d = new Data!(5,2)(testData, flags);

  Normalization norm = Normalization(d.shift, d.scale);

  foreach(i; 0 .. d.nPoints)
  {
    DataPoint!(5,2) tmp = d.getPoint(i);
    norm.unnormalize(tmp);
    assert(approxEqual(tmp.data[], testData[i]));
    norm.normalize(tmp);
    assert(approxEqual(tmp.data[], d.getPoint(i).data[]));
  }

  // Test unnormalizing targets only
  foreach(i; 0 .. d.nPoints)
  {
    double[] tmp = d.getPoint(i).data[($ - 2) .. $];
    norm.unnormalize(tmp);
    assert(approxEqual(tmp, testData[i][($ - 2) .. $]),
      format("\n%s\n%s",tmp,testData[i]));
  }

  // Test again, but this time with an inputs only data set.
  Data!(7, 0) d2 = new Data!(7,0)(testData, flags);
  // Add a few points on the end (that would be for targets, 
  // but they're not in this data set.)
  Normalization norm2 = 
              Normalization(d2.shift[] ~ [1.0, 2.0], d2.scale[] ~ [1.0, 2.0]);

  foreach(i; 0 .. d2.nPoints)
  {
    DataPoint!(7,0) tmp = d2.getPoint(i);
    norm2.unnormalize(tmp);
    assert(approxEqual(tmp.data[], testData[i]));
    norm2.normalize(tmp);
    assert(approxEqual(tmp.data[], d.getPoint(i).data[]));
  }
}

///
unittest
{
  Data!(5, 2) d = new Data!(5,2)(testData, flags);

  Normalization norm = d.normalization;

  foreach(i; 0 .. d.nPoints)
  {
    DataPoint!(5,2) tmp = d.getPoint(i);
    norm.unnormalize(tmp);
    assert(approxEqual(tmp.data[], testData[i]));
    norm.normalize(tmp);
    assert(approxEqual(tmp.data[], d.getPoint(i).data[]));
  }
}
/**
 * Save a normalization to a file so it can be loaded back in later.
 * Useful for associating with a trained network, send the normalizations
 * along with the network file, otherwise the network is useless.
 * 
 * TODO - When phobos library settles down, save these as XML instead.
 *
 * Params: 
 * norm = the normalization to save
 * path = path to the file to save.
 */
void saveNormalization(const Normalization norm, const string path)
{
  /*
   * File format:
   * Normalization
   * nVals = val                      // length of arrays shift and scale
   * shift = val,val,val,val,val,...
   * scale = val,val,val,val,val,...
   */

  assert(norm.scale.length == norm.shift.length);
  
  // Open the file
  File fl = File(path, "w");
  
  // Put a header to identify it as NormalizedData
  fl.writefln("Normalization");

  // The number of elements in the arrays
  size_t nVals = norm.scale.length;
  
  // Save the size
  fl.writefln("nVals = %d",nVals);

  // Save the arrays shift and scale
  fl.writef("shift = ");
  int i;
  for(i = 0; i < nVals - 1; ++i)
    fl.writef("%.*f,", double.dig, norm.shift[i]);
  fl.writef("%.*f\n", double.dig, norm.shift[$ - 1]);

  fl.writef("scale = ");
  for(i = 0; i < nVals - 1; ++i)
    fl.writef("%.*f,", double.dig, norm.scale[i]);
  fl.writef("%.*f\n", double.dig, norm.scale[$ - 1]);
}

/**
 * Load a normalization from the file system.
 *
 * TODO - When phobos library settles down, use XML instead.
 * 
 * Parms: 
 * fileName = path to the file to be loaded.
 */
Normalization loadNormalization(const string fileName)
{
  // See file description in saveNormalization

  // Open the file, read in the contents
  string text = readText(fileName);
  
  // Split into lines
  string[] lines = split(text, regex("\n"));

  // clean up the lines
  foreach(ref line; lines) line = strip(line);
  
  // Parse number of values
  enforce(lines[0] == "Normalization");
  const size_t nVals = to!size_t(lines[1][8 .. $]);

  // Set shift and scale
  double[] tmpShift = new double[](nVals);
  double[] tmpScale = new double[](nVals);

  // Parse shift
  int i;
  string[] tokens = split(lines[2][8 .. $],regex(","));
  for(i = 0; i < nVals; ++i) tmpShift[i] = to!double(strip(tokens[i]));

  // Parse scale
  tokens = split(lines[3][8 .. $],regex(","));
  for(i = 0; i < nVals; ++i) tmpScale[i] = to!double(strip(tokens[i]));

  return Normalization(tmpShift, tmpScale);
}

///
unittest
{
  const Data!(5, 2) d = new Data!(5,2)(testData, flags);

  Normalization norm = d.normalization;

  saveNormalization(norm, "norm.csv");
  scope(exit) std.file.remove("norm.csv");
  
  auto loadedNorm = loadNormalization("norm.csv");
  
  assert(approxEqual(norm.shift,loadedNorm.shift));
  assert(approxEqual(norm.scale,loadedNorm.scale)); 
}
+/
