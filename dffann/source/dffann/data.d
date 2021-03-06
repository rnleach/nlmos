/**
* Utilities for loading, saving, and passing data around.
*
* Author: Ryan Leach
* 
*/
module dffann.data;

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

import numeric;
import dffann;

version(unittest)
{
  import dffann.testutilities.testdata;
}

/*------------------------------------------------------------------------------
*                             DataPoint struct
*-----------------------------------------------------------------------------*/
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
  /// Inputs and targets view.
  public double[] inputs;
  /// ditto
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

  /// ditto
  this(const double[] inpts, const double[] trgts) const
  {
    inputs = inpts;
    targets = trgts;
  }

  /// ditto
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
  enum vals  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
  double[] inpts = vals[0 .. 5];
  double[] trgts = vals[5 .. $];
}

unittest
{
  mixin(announceTest("DataPoint constructor"));

  const DataPoint dp = DataPoint(inpts, trgts);
  assert( dp.inputs == vals[0 .. 5] );

  // DataPoint objects with no targets are possible too.
  const DataPoint dp2 = DataPoint(vals);
  assert( dp2.inputs == vals );
}

unittest
{
  mixin(announceTest("DataPoint toString"));

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
*******************************************************************************/
public class Data
{

  private double[] data_;
  private const uint numInputs;
  private const uint numTargets;
  private const uint numVals;

  /**
  * Basic constructor. This constructor copies the data from the array, so it is
  * not very space efficient for large data sets, but it is safe for immutable
  * data.
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
  public this(in uint nInputs, in uint nTgts, in double[][] data)
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
    this.data_ = new double[](data.length * numVals);
    
    for(size_t i = 0; i < data.length; ++i)
    {
      const size_t start = i * this.numVals;
      assert( data[i].length == this.numVals );
      this.data_[start .. (start + this.numVals)] = data[i][];
    }
  }
  /// ditto
  public this(in uint nInputs, in uint nTgts, in double[][] data) immutable
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
    auto temp = new double[](data.length * numVals);
    
    for(size_t i = 0; i < data.length; ++i)
    {
      const size_t start = i * this.numVals;
      assert( data[i].length == this.numVals );
      temp[start .. (start + this.numVals)] = data[i][];
    }
    this.data_ = cast(immutable) temp;
  }

  /**
  * This constructor just stores a reference to the array data. It was made 
  * private because using it depends on detailed knowledge of the layout of 
  * of the data_ private member. It is intended to be used with helper functions
  * for loading from files. Since it does not make a copy, it is expected to be
  * more space and time efficient, which is important because some of the data
  * sets are expected to be very large.
  *
  * The immutable versions below uses a cast to produce immutable data, and so
  * are very dangerous. Care should be taken to make sure that no other 
  * references to the data argument exist. It also takes a reference to the 
  * data argument so it can destroy that reference by setting it to null, whihc
  * is a start, but still no gaurantee.
  */
  private this(in uint nInputs, in uint nTgts, double[] data)
  {
    this.numInputs  = nInputs;
    this.numTargets = nTgts;
    this.numVals    = nInputs + nTgts;
    this.data_      = data;
  }
  /// ditto
  private this(in uint nInputs, in uint nTgts, ref double[] data) immutable
  {
    this.numInputs  = nInputs;
    this.numTargets = nTgts;
    this.numVals    = nInputs + nTgts;
    this.data_      = cast(immutable double[])data;

    // null data to destroy this reference to the data. Not bullet proof, but a
    // step in the right direction to getting rid of non-immutable references to
    // the underlying data. Of course it might be a suprise to a user of the 
    // constructor....
    data = null;
  }
  
  /**
  * Copy constructor for immutable data. This also copies the data so it can 
  * guarantee that it holds the only reference to the immutable data. No casts
  * here.
  */
  public this(const Data src) immutable
  {
    this.numInputs  = src.numInputs;
    this.numTargets = src.numTargets;
    this.numVals    = src.numVals;
    this.data_      = src.data_.idup;
  }

  /**
  * Get information about the sizes of data associated with this Data object.
  */
  @property final size_t nPoints() const 
  {
    return this.data_.length / this.numVals;
  }

  /// ditto
  @property final size_t nInputs() const { return this.numInputs; }
  
  /// ditto
  @property final size_t nTargets() const { return this.numTargets; }

  /**
  * Returns: a range that iterates over the points in this collection.
  */
  @property final auto simpleRange() const
  {
    return DataRange!(typeof(this))(this);
  }
  /// ditto
  @property final auto simpleRange()
  {
    return DataRange!(typeof(this))(this);
  }

  /**
  * Returns: a range that iterates over the points in this collection in
  *          the same order every time.
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
  public final auto opIndex(size_t index)
  {
    const size_t start = index * numVals;
    const size_t end = start + numVals;
    const size_t brk = start + numInputs;
    return DataPoint(data_[start .. brk], data_[brk .. end]);
  }
  /// ditto
  public final auto opIndex(size_t index) const
  {
    const size_t start = index * numVals;
    const size_t end = start + numVals;
    const size_t brk = start + numInputs;
    return const DataPoint(data_[start .. brk], data_[brk .. end]);
  }
  /// ditto
  public final auto opIndex(size_t index) immutable
  {
    const size_t start = index * numVals;
    const size_t end = start + numVals;
    const size_t brk = start + numInputs;
    return immutable DataPoint(data_[start .. brk], data_[brk .. end]);
  }

  /// Override of $ operator
  public final size_t opDollar(size_t pos)() const 
  {
    return this.data_.length;
  }
}

/*==============================================================================
*                     Unit tests for data class
*=============================================================================*/
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
}

unittest
{ 
  mixin(announceTest("Constructors"));

  Data d = new Data(5, 2, testData);
  
  // Check the number of points
  assert( d.nPoints  == 10 );
  assert( d.nInputs  ==  5 );
  assert( d.nTargets ==  2 );

  foreach( i; 0 .. testData.length)
  {
    assert( d[i] == DataPoint(testData[i][0 .. 5], testData[i][5 .. $]) );
  }
}

/*==============================================================================
*                         Helper Functions for Data class.
*=============================================================================*/
/**
* Load data from a file and create a Data object.
*
* Params:
* nInputs    = number of inputs.
* nTgts      = number of targets.
* filename   = Path to the data to load.
*/
public Data loadDataFromCSVFile(in uint nInputs, in uint nTgts, in string fname)
{ 
  const size_t numVals = nInputs + nTgts;

  // Open the file
  File f = File(fname, "r");

  // Read the file line by line
  auto app = appender!(double[])();
  auto sepRegEx = ctRegex!",";
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
  
  // Return the new Data instance - uses private constructor.
  return new Data(nInputs, nTgts, app.data);
}

unittest
{
  mixin(announceTest("loadDataFromCSVFile"));

  makeRandomCSVFile(100, 22,"TestData.csv");
  scope(exit) std.file.remove("TestData.csv");

  auto d = loadDataFromCSVFile(21, 1, "TestData.csv");

  // Check the number of points
  assert( d.nPoints  == 100 );
  assert( d.nInputs  ==  21 );
  assert( d.nTargets ==   1 );
}

/** 
* Save data in a pre-determined format so that it  can be loaded quickly. 
* 
* Params: 
* pData    = The data object to save.
* filename = The path to save the data.
*/
public void saveData(const Data pData, const string filename)
{
  /*
  * File format:
  * Data
  * nPoints = val
  * nInputs = val
  * nTargets = val
  * data =
  * inputVal,inputVal,inputVal,inputVal,.....targetVal,targetVal,...
  */

  // Open the file
  File fl = File(filename, "w");

  // Put a header to identify it as NormalizedData
  fl.writefln("Data");

  // Save the variables
  fl.writefln("nPoints = %d",pData.nPoints);
  fl.writefln("nInputs = %d",pData.nInputs);
  fl.writefln("nTargets = %d",pData.nTargets);

  // Now write the data points
  fl.writefln("data = ");
  foreach(dp; pData.simpleRange)
    fl.writefln(dp.stringRep);
}

/** 
* Load data that was saved via saveData.
* 
* Params: 
* filename = The path to the file to load.
* 
* Returns: A Data object.
*/
public auto loadData(string opt = "immutable")(const string filename)
  if( opt == "immutable" || opt == "mutable")
{
  // See comments in saveData for file and header formats.

  // Open the file, read in the contents
  string text = readText(filename);

  // Split into lines as header and data sections
  string[] lines = split(text, regex("\n"));
  string[] header = lines[0 .. 5];
  string[] dataLines = lines[5 .. $];

  // clean up the header lines
  foreach(ref line; header) line = strip(line);

  // Parse some variables out of the header section
  const size_t nmPoints = to!size_t(header[1][10 .. $]);
  const uint nmInputs = to!uint(header[2][10 .. $]);
  const uint nmTargets = to!uint(header[3][11 .. $]);
  const size_t numVals = nmInputs + nmTargets;

  // Check that this file matches the kind of Data object we are loading.
  enforce(header[0] == "Data");

  // set up some variables, define nested function for mixin to
  // parse arrays.
  string[] tokens;
  
  // Now parse each row
  enforce(dataLines.length >= nmPoints,
    "Malformed data file, not enough input points.");

  double[] tmp = new double[](nmPoints * numVals);
  
  foreach(i; 0 ..nmPoints)
  {
    tokens = split(dataLines[i], regex(","));

    enforce(tokens.length >= numVals,
      "Malformed data file, not enought points on line.");

    const start = i * numVals;
    foreach(j; start .. (start + numVals))
      tmp[j] = to!double(strip(tokens[j - start]));

  }

  static if(opt == "immutable")
  {
    return new immutable(Data)( nmInputs, nmTargets, tmp);
  }
  else static if(opt == "mutable")
  {
    return new Data( nmInputs, nmTargets, tmp);
  }
  else static assert(0, "Invalid template arguement parameter.");
}

unittest
{
  mixin(announceTest("loadData"));

  enum string testFileName = "TestData.csv";
  enum string testSaveName = "TestSaveData.csv"; 
  
  makeRandomCSVFile(222, 22, testFileName);
  scope(exit) std.file.remove(testFileName);

  Data d = loadDataFromCSVFile(21, 1, testFileName);
  
  saveData(d, testSaveName);
  scope(exit) std.file.remove(testSaveName);
  
  auto d2 = loadData(testSaveName);

  assert( d2.nPoints == d.nPoints   );
  assert( d2.nInputs == d.nInputs   );
  assert( d2.nTargets == d.nTargets );
  
  foreach(k; 0 .. d.nPoints)
  {
    assert( approxEqual(d2[k].inputs, d[k].inputs) );
    assert( approxEqual(d2[k].targets, d[k].targets) );
  }
}
/*==============================================================================
*                                   DataRange
*=============================================================================*/
/**
* RandomAccessRange with length and slicing for iterating over a Data object.
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

  /// Copy constructor
  this(DataRange!DataType src)
  {
    this.start_ = src.start_;
    this.end_ = src.end_;
    this.dataObject_ = src.dataObject_;
  }

  void opAssign(DataRange!DataType rhs)
  {
    enforce(rhs.dataObject_ is this.dataObject_, 
      "Cannot assign slices between different data sources.");
    
    this.start_ = rhs.start_;
    this.end_ = rhs.end_;
  }
  
  /// Properties/methods to make this a RandomAccessRange.
  @property bool empty() { return start_ >= end_; }

  /// ditto
  @property auto ref front() { return this.dataObject_[start_]; }

  /// ditto
  void popFront() { ++start_; }

  /// ditto
  @property auto ref back() { return this.dataObject_[end_ - 1]; }

  /// ditto
  void popBack() { --end_; }

  /// ditto
  // Since this is a struct, return copies all values! Easy.
  @property auto save() { return this; }

  auto ref opIndex(size_t index)
  {
    return this.dataObject_[start_ + index];
  }

  /// ditto
  @property size_t length(){return end_ - start_;}

  /**
  * This is nice to have, but only works for assignment if the range you are
  * assigning to has the same underlying data object. This condition is checked
  * for with std.exception.enforce.
  *
  * Examples:
  * ------------
  * auto d1 = new immutable(Data)(5,2, src1);
  * auto d2 = new immutable(Data)(5,2, src2);
  *
  * auto r1 = d1.simpleRange;  // Initialization, not assignment
  * auto r2 = d2.simpleRange;  // Initialization, not assignment
  * auto r3 = r1;              // OK, initialization, not assignment
  * r1 = r1[2 .. $];           // OK, underlying data is the same, d1.
  * r3 = r1[3 .. $];           // OK, underlying data is the same, d1.
  * r2 = r2[1 .. $];           // OK, underlying data is the same, d2.
  * r2 = r1[2 .. $];           // ERROR, underlying data is different, d2 !is d1
  * ------------
  */
  typeof(this) opSlice(size_t start, size_t end)
  {
    assert(start <= end, "Range Error, start must be less than end.");
    assert(start >= 0, "No negative indicies allowed!");
    assert( this.start_ + end <= this.end_, "Out of range!" );

    // Create a copy
    auto temp = DataRange!DataType(this.dataObject_);

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
static assert( isInputRange!(DataRange!(immutable Data))         );
static assert( isForwardRange!(DataRange!(immutable Data))       );
static assert( isBidirectionalRange!(DataRange!(immutable Data)) );
static assert( isRandomAccessRange!(DataRange!(immutable Data))  );
static assert( hasLength!(DataRange!(immutable Data))            );
static assert( hasSlicing!(DataRange!(immutable Data))           ); 

unittest
{
  mixin(announceTest("DataRange"));

  // Create a data set to test on.
  auto d = new immutable(Data)(5, 2, testData);

  alias DR = DataRange!(typeof(d));

  // Create a range to test
  auto r = DR(d);
  auto r2 = r.save;
  auto r3 = r.save;

  // Check length
  assert(testData.length == r.length);

  // Test that the elements line up.
  size_t i = 0;
  foreach(t; r) assert( t == d[i++] );
  
  i = d.nPoints - 1;
  while(!r2.empty)
  {
    auto t = r2.back;
    r2.popBack();

    assert( t == d[i] );
    assert( r2.length == i );
    i--;
  }
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
  
  /// Properties/methods to make this an infinite ForwardRange.
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

unittest
{
  mixin(announceTest("RandomDataRange"));

  // Create a data set to test on.
  auto d = new immutable(Data)(5, 2, testData);

  alias DR = RandomDataRange!(typeof(d));

  // Create a range to test
  auto r = DR(d);

  // Test if take works -  if no exceptions, call it good for now. Can make a
  // better test for this.
  auto r2 = take(r, d.nPoints * 100);
}

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

unittest
{
  mixin(announceTest("InfiniteDataRange"));

  // Create a data set to test on.
  Data d = new Data(5, 2, testData);

  alias DR = InfiniteDataRange!(typeof(d));

  // Create a range to test
  auto r = DR(d);

  size_t i = 0;
  foreach(t; take(r, d.nPoints * 100))
  {
    size_t idx = i % d.nPoints;
    assert( t == d[idx] );
    i++;
  }
}

/*==============================================================================
*                                Normalizations
*=============================================================================*/

/**
* A normalization is used to force data to have certain statistical properties,
* such as a standard deviation of 1 and a mean of 0. 
* 
* Sometimes data is binary, e.g. with values 0 or 1, or possibly -1 or 1, in 
* which case you do not want to normalize it. Binary data may also be mixed 
* with non-binary data. This is handled at the constructor level when loading
* the data with the use of filters to determine if a column is binary.
*/
struct Normalization
{

  public const double[] shift;
  public const double[] scale;
  
  /**
  * Params: 
  * shift = array of shifts to subtract from each point to be normalized.
  * scale = array of scales to divide each point by.
  */
  public this(const double[] shift, const double[] scale)
  {
    enforce(shift.length == scale.length, 
      "Shift and scale array lengths differ.");

    this.shift = shift.dup;
    this.scale = scale.dup;
  }

  /**
  * Normalizes the Data dt in place.
  */
  public void normalizeInPlace(Data dt)
  {
    assert( dt.numVals == shift.length );

    const double[] inShift = shift[0 .. dt.nInputs];
    const double[] inScale = scale[0 .. dt.nInputs];
    const double[] tgtShift = shift[dt.nInputs .. dt.numVals];
    const double[] tgtScale = scale[dt.nInputs .. dt.numVals];

    auto rng = dt.simpleRange;

    foreach(dp; rng)
    {
      dp.inputs[] = (dp.inputs[] - inShift[]) / inScale[];
      dp.targets[] = (dp.targets[] - tgtShift[]) / tgtScale[];
    }
  }

  /**
  *  Un-normalize a single point with nTgtVals number of target values.
  */
  public void unNormalizeTarget(double[] tgt)
  {
    tgt[] = tgt[] * scale[($ - tgt.length) .. $] + shift[($ - tgt.length) .. $];
  }
}

version(unittest)
{
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

  enum double[] shiftPar = [1.45, 2.45, 3.45, 4.45, 5.45, 6.45, 7.45];
  double[] scalePar = new double[](shiftPar.length);
  double scaleVal = 0.302765035409750;
}

unittest
{
  mixin(announceTest("Normalization."));

  Data d = new Data(5, 2, testData);
  scalePar[] = scaleVal;

  Normalization norm = Normalization(shiftPar, scalePar);

  norm.normalizeInPlace(d);

  size_t i = 0;
  foreach(dp; d.simpleRange)
  {
    foreach(ival; dp.inputs)
    {
      assert( approxEqual(ival, normalizedRowValues[i]) );
    }
    foreach(ival; dp.targets)
    {
      assert( approxEqual(ival, normalizedRowValues[i]) );
    }
    i++;
  }

  foreach(dp; d.simpleRange)
  {
    norm.unNormalizeTarget(dp.targets);
  }

  i = 0;
  foreach(dp; d.simpleRange)
  {
    assert( approxEqual(dp.targets, testData[i][5 .. $]) );
    i++;
  }
}

/**
* Calculate a normalization given the data set.
*/
Normalization calcNormalization(const Data dt, const bool[] binaryFilter)
{
  assert( binaryFilter.length == dt.numVals );

  const size_t numVals = dt.nInputs + dt.nTargets;
  const size_t nInputs = dt.nInputs;
  const size_t nTargets = dt.nTargets;

  /*==========================================================================
    Nested struct to hold results of summing over a batch
  ==========================================================================*/
  struct BatchResults 
  {
    public double[] batchSum;
    public double[] batchSumSquares;

    public this(double[] sum, double[] sumSq)
    {
      this.batchSum = sum;
      this.batchSumSquares = sumSq;
    }
  }

  /*==========================================================================
    Nested function to calculate a batch of stats using Kahan algorithm.
  ==========================================================================*/
  BatchResults sumChunk(DR)(DR chunk)
  {
    /*
    This could be much more efficient probably, but we shouldn't do this often.
    */
    double[] sm = new double[](numVals);
    double[] smc = new double[](numVals);
    double[] smSq = new double[](numVals);
    double[] smSqc = new double[](numVals);
    sm[] = 0.0;
    smc[] = 0.0;
    smSq[] = 0.0;
    smSqc[] = 0.0;

    double[] yIn = new double[](nInputs);
    double[] tIn = new double[](nInputs);
    double[] yT = new double[](nTargets);
    double[] tT = new double[](nTargets);

    // Calculate the sum and sumsq
    foreach(d; chunk)
    {
      yIn[] = d.inputs[] - smc[0 .. nInputs];
      tIn[] = sm[0 .. nInputs] + yIn[];
      smc[0 .. nInputs] = (tIn[] - sm[0 .. nInputs]) - yIn[];
      sm[0 .. nInputs] = tIn[];

      yT[] = d.targets[] - smc[nInputs .. numVals];
      tT[] = sm[nInputs .. numVals] + yT[];
      smc[nInputs .. numVals] = (tT[] - sm[nInputs .. numVals]) - yT[];
      sm[nInputs .. numVals] = tT[];

      yIn[] = d.inputs[] * d.inputs[] - smSqc[0 .. nInputs];
      tIn[] = smSq[0 .. nInputs] + yIn[];
      smSqc[0 .. nInputs] = (tIn[] - smSq[0 .. nInputs]) - yIn[];
      smSq[0 .. nInputs] = tIn[];

      yT[] = d.targets[] * d.targets[] - smSqc[nInputs .. numVals];
      tT[] = smSq[nInputs .. numVals] + yT[];
      smSqc[nInputs .. numVals] = (tT[] - smSq[nInputs .. numVals]) - yT[];
      smSq[nInputs .. numVals] = tT[];
    }

    return BatchResults(sm, smSq);
  }

  /*==========================================================================
    Now do the summations in parallel.
  ==========================================================================*/
  double[] sum = new double[](numVals);
  double[] sumsq = new double[](numVals);
  double[] shift = new double[](numVals);
  double[] scale = new double[](numVals);

  // How many threads to use?
  size_t numThreads = totalCPUs - 1;
  if(numThreads < 1) numThreads = 1;

  BatchResults[] reses = new BatchResults[numThreads];
  
  auto drs = evenChunks(dt.simpleRange, numThreads);

  alias Rng = typeof(drs.front);
  Rng[] dra = array(drs);

  foreach(i, ref res; parallel(reses))
  {
    res = sumChunk(dra[i]);
  }

  // Initialize
  shift[] = 0.0;
  scale[] = 1.0;
  sum[] = 0.0;
  sumsq[] = 0.0;
  
  // For Kahan summation
  double[] sumc = new double[](numVals);
  double[] sumsqc = new double[](numVals);
  double[] y = new double[](numVals);
  double[] t = new double[](numVals);
  sumc[] = 0.0;
  sumsqc[] = 0.0;

  foreach(res; reses)
  {
    y[] = res.batchSum[] - sumc[];
    t[] = sum[] + y[];
    sumc[] = (t[] - sum[]) - y[];
    sum[] = t[];

    y[] = res.batchSumSquares[] - sumsqc[];
    t[] = sumsq[] + y[];
    sumsqc[] = (t[] - sumsq[]) - y[];
    sumsq[] = t[];
  }
  
  // Calculate the mean (shift) and standard deviation (scale)
  size_t nPoints = dt.nPoints;
  for(size_t i = 0; i < numVals; ++i)
  {
    shift[i] = sum[i] / nPoints;
    scale[i] = sqrt((sumsq[i] / nPoints - shift[i] * shift[i]) * 
      nPoints / (nPoints - 1));
  }
  
  // All done, now return.
  return Normalization(shift, scale);
}

unittest
{
  mixin(announceTest("calcNormalization"));

  // None of these values are binary, so all flags are false
  bool[] flags = [false, false, false, false, false, false, false]; 

  Data d = new Data(5, 2, testData);
  Normalization norm = calcNormalization(d, flags);

  norm.normalizeInPlace(d);

  size_t i = 0;
  foreach(dp; d.simpleRange)
  {
    foreach(ival; dp.inputs)
    {
      assert( approxEqual(ival, normalizedRowValues[i]) );
    }
    foreach(ival; dp.targets)
    {
      assert( approxEqual(ival, normalizedRowValues[i]) );
    }
    i++;
  }

  foreach(dp; d.simpleRange)
  {
    norm.unNormalizeTarget(dp.targets);
  }

  i = 0;
  foreach(dp; d.simpleRange)
  {
    assert( approxEqual(dp.targets, testData[i][5 .. $]) );
    i++;
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

unittest
{
  mixin(announceTest("saveNormalization - loadNormalization"));

  scalePar[] = scaleVal;

  Normalization norm = Normalization(shiftPar, scalePar);

  saveNormalization(norm, "norm.csv");
  scope(exit) std.file.remove("norm.csv");
  
  auto loadedNorm = loadNormalization("norm.csv");
  
  assert(approxEqual(norm.shift,loadedNorm.shift));
  assert(approxEqual(norm.scale,loadedNorm.scale)); 
}
