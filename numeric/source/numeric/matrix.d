/**
* Author: Ryan Leach
* Version: 1.0.0
* Date: January 15, 2015
*/
module numeric.matrix;

import std.algorithm: max, min;
import std.array: appender, uninitializedArray;
import std.exception: enforce;
import std.math;
import std.random;
import std.string: format;

import std.experimental.allocator: IAllocator, processAllocator, makeArray, dispose;

import numeric.numeric;

version(par)
{
  import std.parallelism: parallel;
}

/**
* Matrix struct for basic matrix operations.
*/
public struct Matrix
{
  /**
  * The current memory allocator, which can be changed. Instances keep a 
  * reference to the allocator they were created with, so this can be changed
  * a anytime without affecting already created objects. Any new matrices will
  * be allocated classAlloc.
  */
  public static IAllocator classAlloc;

  // Remember how my array was allocated. 
  private IAllocator localAlloc_;

  /*
  * Store the matrix internally in a singly allocated chunk of memory.
  * Then the postblit constructor will have to manually take care of moving
  * everything.
  */
  private size_t rows;
  private size_t cols;
  private size_t numVals;  // numVals = rows * cols is used a lot, so cache it!
  package double[] m;

  // Initialize static allocator variable at runtime.
  static this() { classAlloc = processAllocator; }

  /*============================================================================
  *                Memory management, constructors, destructor, etc.
  *===========================================================================*/
  /**
  * Basic constructor, create uninitialized matrix.
  *
  * The single variable versions create square Matrices.
  */
  this(in size_t r, in size_t c)
  {
    localAlloc_ = classAlloc;

    rows = r;
    cols = c;
    numVals = rows * cols;

    m = localAlloc_.makeArray!double(numVals);
  }

  /// ditto
  this(in size_t r) { this(r,r); }

  /// Initialize from an array
  this(in double[][] arr)
  {
    localAlloc_ = classAlloc;

    // Get the dimensions
    const size_t r = arr.length;
    const size_t c = arr[0].length;

    // Make sure the array is not a jagged array
    foreach(arrRow; arr) enforce(arrRow.length == c, "Non-rectangular array.");

    // Now set up the Matrix
    rows = r;
    cols = c;
    numVals = rows * cols;

    // Allocate memory on the heap (most likely, may be on stack....)
    m = localAlloc_.makeArray!double(numVals);

    // Copy in values
    foreach(i; 0 .. r)
    {
      const rowStart = i * c;
      m[rowStart .. (rowStart + c)] = arr[i][];
    }
  }

  /// Initialize with a specific value
  this(in size_t r, in size_t c, in double initVal)
  {
    localAlloc_ = classAlloc;

    rows = r;
    cols = c;
    numVals = r * c;

    m = localAlloc_.makeArray!double(numVals, cast()initVal);
  }

  /// Copy constructor
  this(in Matrix orig)
  {
    // Use the latest and greatest in allocators
    localAlloc_ = classAlloc;

    rows = orig.rows;
    cols = orig.cols;
    numVals = orig.numVals;

    // Duplicate array data, this is a deep copy
    m = localAlloc_.makeArray!double(orig.m);
  }

  unittest
  {
    mixin(announceTest("Constructors"));

    // Test blank initialization
    Matrix M = Matrix(4,5);
    assert(M.rows == 4);
    assert(M.cols == 5);
    assert(M.numVals == 20);

    // Test square matrix initialization
    M = Matrix(4);
    assert(M.rows == 4);
    assert(M.cols == 4);
    assert(M.numVals == 16);

    // Test initialization from an array    
    double[][] arr = [[0.0, 1.0, 2.0],
                      [0.1, 1.1, 2.1],
                      [0.2, 1.2, 2.2]];

    Matrix N = Matrix(arr);
    assert(N[0,0] == 0.0);
    assert(N[0,1] == 1.0);
    assert(N[0,2] == 2.0);
    assert(N[1,0] == 0.1);
    assert(N[1,1] == 1.1);
    assert(N[1,2] == 2.1);
    assert(N[2,0] == 0.2);
    assert(N[2,1] == 1.2);
    assert(N[2,2] == 2.2);
  }

  /// Postblit constructor.
  this(this) 
  {
    // Update to the current default allocator
    localAlloc_ = classAlloc;

    // rows and cols were moved for us, now we have to allocate new memory and
    // copy all the values to them
    m = localAlloc_.makeArray!double(m);
  }

  ~this() 
  { 
    //writef("In destructor with m = %s...", m);stdout.flush();
    if(m) localAlloc_.dispose(m); 
    //writefln("Exiting destructor with m = %s", m);stdout.flush();
  }

  unittest
  {
    mixin(announceTest("Postblit"));

    Matrix M = Matrix(4, 5);

    // Fill the array with something other than double.nan
    foreach(size_t i; 0 .. M.numVals) M.m[i] = uniform!"[]"(0.0, 1.0);

    // Run the postblit
    Matrix N = M;
    assert( N !is M);    // Not the same address anymore

    // Now test each element in each is the same
    foreach(size_t i; 0 .. M.numVals) assert(N.m[i] == M.m[i]);

    // Demonstrate assignment to 1 does not affect the other.
    N.m[0] = 3.14159;
    foreach(size_t i; 0 .. M.numVals)
    {
      if(i == 0)
      { 
        assert(M.m[i] != N.m[i]);
        assert(N.m[i] == 3.14159);
      }
      else assert(N.m[i] == M.m[i]);
    }
  }

  /// Make a duplicate matrix.
  @property Matrix dup() const { return Matrix(this); }

  /**
  * Make a human readable string representation of a matrix suitable for
  * terminal output. Not a good idea to use this on large Matrices.
  */
  string toString() const 
  {
    auto app = appender!string();

    // 100 for labels etc, numVals * 8 for the numbers, rows for the newlines
    app.reserve(100 + numVals * 8 + rows);
    foreach(r; 0 .. rows)
    {
      const size_t rowStart = r * cols;
      foreach(c; 0 .. cols)
      {
        app ~= format("%6f  ",m[rowStart + c]);
      }
      app ~= "\n";
    }
    app ~= "\n";

    //return toRet;
    return app.data;
  }

  /*============================================================================
  *                        Static factory methods
  *===========================================================================*/
  /// Create a matrix initialized to any desired value.
  static Matrix matrixOf(in double val, in size_t r, in size_t c)
  {
    return Matrix(r, c, val);
  }
  /// Create a square matrix initialized to any desired value.
  static Matrix matrixOf(in double val, in size_t dim)
  {
    return Matrix(dim, dim, val);
  }

  unittest
  {
    mixin(announceTest("matrixOf"));

    Matrix M = Matrix.matrixOf(2.17,7,8);
    foreach(i; 0 .. M.numVals) assert(M.m[i] == 2.17);
  }

  /// Create a matrix of zeros.
  static Matrix zeros(in size_t r, in size_t c)
  {
    return Matrix(r, c, 0.0);
  }
  
  /// ditto
  static Matrix zeros(in size_t r){ return zeros(r,r); }

  unittest
  {
    mixin(announceTest("Matrix.zeros")); 
    Matrix M = Matrix.zeros(3,5);
    foreach(i; 0 .. M.numVals) assert(M.m[i] == 0.0);
  }
  
  /// Create an identity matrix.
  static Matrix identity(in size_t r)
  {
    Matrix temp = zeros(r);
    // Make the diagonal 1.0s
    foreach(i; 0 .. r) temp.m[i * r + i] = 1.0;

    return temp;
  }

  unittest
  {
    mixin(announceTest("Matrix.identity"));

    Matrix M = Matrix.identity(3);

    foreach(r; 0 .. M.numRows)
      foreach(c; 0 .. M.numCols)
        if(r == c) assert(M.m[r * M.numCols + c] == 1.0);
        else assert(M.m[r * M.numCols + c] == 0.0);
  }
  
  /**
  * Create a matrix filled with random numbers. 
  * 
  * This may later be expanded to fully leverage the distribution functions 
  * and generators in std.random. For now it just produces a matrix with 
  * random double values from [0.0,1.0] inclusive.
  */
  static Matrix random(in size_t r, in size_t c)
  {
    Matrix temp = Matrix(r,c);

    foreach(i; 0 .. temp.numVals) temp.m[i] = uniform!"[]"(0.0, 1.0);

    return temp;
  }
  /// ditto
  static Matrix random(in size_t r){return Matrix.random(r,r);}

  /*============================================================================
  *                   Index Operators, Limits, and Assignment
  *===========================================================================*/
  /**
  * Get a value from the Matrix.
  *
  * Examples:
  * ---------
  * Matrix M = Matrix.random(4);
  * double val = M[1,2];
  * ---------
  *
  * Params: 
  * r = the row number
  * c = the column number
  *
  */
  double opIndex(in size_t r, in size_t c) const
  {
    // Bounds check, this goes away with release builds
    assert(r < rows && c < cols,"Index out of bounds error.");

    return m[r * cols + c];
  }
  unittest
  {
    mixin(announceTest("opIndex"));

    Matrix M = matrixOf(4.0, 3, 3);

    assert(M[0, 0] == 4.0);
    assert(M[0, 1] == 4.0);
    assert(M[0, 2] == 4.0);
    assert(M[1, 0] == 4.0);
    assert(M[1, 1] == 4.0);
    assert(M[1, 2] == 4.0);
    assert(M[2, 0] == 4.0);
    assert(M[2, 1] == 4.0);
    assert(M[2, 2] == 4.0);

    // No bounds checking, so no assertions for errors. The assert statement
    // in the first line is only present for debugging code, code compiled in
    // release mode (or optomize) will not have this assertion.
  }

  /**
  * Set the value of a cell in the matrix.
  *
  * Examples:
  * ---------
  * Matrix M = Matrix.random(4);
  * M[1,2] = 3.0;
  * ---------
  *
  * Params: 
  * val = the set the matrix cell to.
  * r   = the row number
  * c   = the column number
  */
  void opIndexAssign(in double val, in size_t r, in size_t c)
  {
    // Bounds check, this goes away with release builds
    assert(r < rows && c < cols,"Index out of bounds error.");

    m[r * cols + c] = val;
  }

  unittest
  {
    mixin(announceTest("opIndexAssign"));

    Matrix M = Matrix(3);
    foreach(r; 0 .. M.numRows)
      foreach(c; 0 .. M.numCols)
      {
        M[r,c] = r * c;
        assert(M[r,c] == r * c);
      }
  }

  /**
  * Set the value of a cell in the matrix with an operation.
  *
  * Examples:
  * ---------
  * Matrix M = Matrix.random(4);
  * 
  * M[1,2] += 1.0; 
  * M[0,0] *= 2.0;
  * M[3,3] /= 15.0;
  * M[2,1] -= 9.81;
  * ---------
  *
  * Params:
  * val = the set the matrix cell to.
  * r   = the row number
  * c   = the column number
  *
  */
  void opIndexOpAssign(string op)(in double val, in size_t r, in size_t c)
  if(op == "+" || op == "-" || op == "*" || op == "/")
  {
    // Bounds check, this goes away with release builds
    assert(r < rows && c < cols,"Index out of bounds error.");

    mixin("m[r * cols + c] " ~ op ~ "= val;");
  }

  unittest
  {
    mixin(announceTest("opIndexOpAssign"));

    Matrix M = Matrix.matrixOf(1.0,3,3);
    foreach(r; 0 .. M.numRows)
      foreach(c; 0 .. M.numCols){

        double val = 1.0;
        
        M[r,c] += r * c; 
        val += r * c;
        assert(M[r,c] == val);
        
        M[r,c] -= 2.0 * c; 
        val -= 2.0 * c;
        assert(M[r,c] == val);
        
        M[r,c] *= 2.0; 
        val *= 2.0;
        assert(M[r,c] == val);
        
        M[r,c] /= 10.0; 
        val /= 10.0;
        assert(M[r,c] == val);
      }
  }

  /// Returns: The number of rows in this matrix.
  @property size_t numRows() { return rows; }

  /// Returns: The number of columns in this matrix.
  @property size_t numCols() { return cols; }

  unittest
  {
    mixin(announceTest("numRows, numCols"));

    Matrix M = Matrix(800,500);
    assert(M.numRows == 800);
    assert(M.numCols == 500);
  }

  /*============================================================================
  *                           Other operators
  *==========================================================================*/
  /**
  * Matrix comparison.
  *
  * Examples:
  * ---------
  * Matrix A = Matrix.matrixOf(1.0, 3, 9);
  * Matrix B = Matrix.matrixOf(1.0, 3, 9);
  * Matrix C = Matrix.matrixOf(0.0, 2, 2);
  *
  * A == B; // true
  * A == C; // false
  * A != C; // true
  *
  * assert(A == B && A != C); // Ok!
  * ---------
  */
  bool opEquals(in Matrix rhs) const
  {
    // Test for same object first.
    if(this is rhs) return true;

    // Now compare them
    if(rows == rhs.rows && cols == rhs.cols)
      return m == rhs.m;
    
    else return false;
  }

  unittest
  {
    mixin(announceTest("opEquals"));

    Matrix M = Matrix.matrixOf(3.14159, 80, 4);
    Matrix N = M; // Copied via postblit
    Matrix O = Matrix.matrixOf(1.0, 80, 4);
    Matrix P = Matrix.matrixOf(3.14159, 79, 4);

    assert(M is M);  // Duh
    assert(M !is N); // Check that it is a copy, so at different address
    assert(M == M);  // Duh
    assert(M == N);  // Now do the reall comparison
    assert(M != O);  // Check matrix with same dimensions, but different values.
    assert(M != P);  // Check matrix with different dimensions and same values.
    assert(O != P);  // The rest of these are for good measure.
    assert(N != O);
    assert(N != P);
  }

  /**
  * Negation operator. 
  *
  * Examples:
  * ---------
  * Matrix A = Matrix.matrixOf(3.0, 2, 2);
  * Matrix B = -A;
  * 
  * assert(B[0,0] == -3.0);
  * assert(B[0,1] == -3.0);
  * assert(B[1,0] == -3.0);
  * assert(B[1,1] == -A[1,1]);
  * ---------
  */
  Matrix opUnary(string op)() const
    if(op == "-")
  {
    Matrix temp = cast(Matrix)this;

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        size_t rw = r * cols;
        temp.m[rw .. (rw + cols)] = -temp.m[rw .. (rw + cols)];
      }
    }
    else{
      temp.m[] = -temp.m[];
    }
    return temp;
  }
  unittest
  {
    mixin(announceTest("opUnary"));

    double[][] m = [[1.0, 2.0],
                    [3.0, 4.0]];
    double[][] n = [[-1.0, -2.0],
                    [-3.0, -4.0]];

    Matrix M = Matrix(m);
    Matrix N = Matrix(n);
    Matrix O = -M;

    assert(N == O);

    // Check to make sure we didn't overwrite M
    assert(M[0,0] == 1.0);
    assert(M[0,1] == 2.0);
    assert(M[1,0] == 3.0);
    assert(M[1,1] == 4.0);

    assert(O[0,0] == -1.0);
    assert(O[0,1] == -2.0);
    assert(O[1,0] == -3.0);
    assert(O[1,1] == -4.0);
  }

  /*============================================================================
  *                  Addition/Subtraction Operator Overloads
  *===========================================================================*/
  /// Overloaded addition and subtraction operators.
  Matrix opBinary(string op)(in Matrix rhs) const
    if( op == "+" || op == "-") 
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(rows == rhs.rows && cols == rhs.cols);

    // Make a new matrix
    Matrix temp = Matrix(rows,cols);

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        size_t rw = r * cols;
        mixin("temp.m[rw .. (rw + cols)] = "
          "m[rw .. (rw + cols)] " ~ op ~ " rhs.m[rw .. (rw + cols)];");
      }
    }
    else
    {
      // Now do the element by element op
      temp.m[] = mixin("m[] " ~ op ~ " rhs.m[]");
    }
    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary + - "));

    // Test addition
    Matrix M = Matrix.identity(3);
    Matrix N = Matrix.matrixOf(1.0,3,3);
    Matrix O = M + N;

    foreach(size_t i; 0 .. O.numRows)
      foreach(size_t j; 0 .. O.numCols)
        if(i == j) assert(O[i,j] == (1.0 + 1.0));
        else assert(O[i,j] == (0.0 + 1.0));

    // Test subtraction
    O = M - N;
    foreach(size_t i; 0 .. O.numRows)
      foreach(size_t j; 0 .. O.numCols)
        if(i == j) assert(O[i,j] == (1.0 - 1.0));
        else assert(O[i,j] == (0.0 - 1.0));
  }

  /// Overloaded addition and subtraction operators with assignment.
  ref Matrix opOpAssign(string op)(in Matrix rhs) 
    if( op == "+" || op == "-")
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(rows == rhs.rows && cols == rhs.cols);

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        size_t rw = r * cols;
        mixin("m[rw .. (rw + cols)] " ~ op ~ "= rhs.m[rw .. (rw + cols)];");
      }
    }
    else{
      // Now do the element by element op
      mixin("m[] " ~ op ~ "= rhs.m[];");
    }

    return this;
  }

  unittest
  {
    mixin(announceTest("opOpAssign + - "));

    // Test addition
    Matrix M = Matrix.identity(3);
    Matrix N = Matrix.matrixOf(1.0,3,3);
    M += N;
    foreach(size_t i; 0 .. M.numRows)
      foreach(size_t j; 0 .. M.numCols)
        if(i == j) assert(M[i,j] == (1.0 + 1.0));
        else assert(M[i,j] == (0.0 + 1.0));

    // Test subtraction
    M -= N;
    foreach(size_t i; 0 .. M.numRows)
      foreach(size_t j; 0 .. M.numCols)
        if(i == j) assert(M[i,j] == (1.0 + 1.0 - 1.0));
        else assert(M[i,j] == (0.0 + 1.0 - 1.0));
  }
  
  /// Overloaded addition and subtraction operators with assignment.
  ref Matrix opOpAssign(string op)(in ref TransposeView rhs) 
    if( op == "+" || op == "-")
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(rows == rhs.src.cols && cols == rhs.src.rows);

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
        foreach(c; 0 .. cols)
          mixin("m[r * cols + c] " ~ op ~ "= rhs.src.m[c * rhs.src.cols + r];");
    }
    else
    {
      // Now do the element by element op
      foreach(r; 0 .. rows)
        foreach(c; 0 .. cols)
          mixin("m[r * cols + c] " ~ op ~ "= rhs.src.m[c * rhs.src.cols + r];");
    }

    return this;
  }

  unittest
  {
    mixin(announceTest("opOpAssign + - TransposeView"));

    double[][] m = [[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]];
    
    double[][] n = [[1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0]];
    
    double[][] o = [[2.0, 5.0, 8.0],
                    [6.0, 9.0,12.0]];
                    
    Matrix M = Matrix(m);
    TransposeView MTv = M.Tv;
    
    Matrix N = Matrix(n);
    TransposeView NTv = N.Tv;
    
    Matrix O = Matrix(o);
    Matrix OT = O.T;
    
    M += NTv;
    foreach(r; 0 .. M.rows)
      foreach(c; 0 .. M.cols)
        assert(M[r,c] == O[r,c]);
    assert(M == O);
    
    M = Matrix(m);
    N += MTv;
    foreach(r; 0 .. N.rows)
      foreach(c; 0 .. N.cols)
        assert(N[r,c] == OT[r,c]);
    assert(N == OT);
    
    OT -= MTv;
    N = Matrix(n);
    foreach(r; 0 .. OT.rows)
      foreach(c; 0 .. OT.cols)
        assert(OT[r,c] == N[r,c]);
    assert(OT == N);
  }

  /*============================================================================
  *                Multiplication and Division by scalars
  *===========================================================================*/
  /**
  * Overloaded multiplication and divistion operators for operations with a
  * Matrix and scalar.
  */
  Matrix opBinary(string op)(in double rhs) const
    if( op == "*" || op == "/")
  {
    // Make a new matrix
    Matrix temp = Matrix(rows,cols);

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        size_t rw = r * cols;
        mixin("temp.m[rw .. (rw + cols)] = m[rw .. (rw + cols)]" ~ op ~ "rhs;");
      }
    }
    else
    {
      // Now do the element by element op
      mixin("temp.m[] = m[] " ~ op ~ " rhs;");
    }

    return temp;
  }

  /**
  * Overloaded multiplication and division operators for operations with a
  * Matrix and scalar. Only allow Multiplication from the left, it is
  * undefined what scalar / Matrix is.
  */
  Matrix opBinaryRight(string op)( in double lhs) const
    if( op == "*")
  {
    // Commutative property of multiplication for scalars.
    return opBinary!op(lhs);
  }

  unittest
  {
    mixin(announceTest("opBinary * / double"));

    // Test multiplication
    Matrix M = Matrix.matrixOf(1.0,3,3);
    Matrix N = M * 2.0;
    foreach(size_t i; 0 .. N.numRows)
      foreach(size_t j; 0 .. N.numCols)
        assert(N[i,j] == (1.0 * 2.0));
    // Test opBinaryRight version
    N = 2.0 * N;
    foreach(size_t i; 0 .. N.numRows)
      foreach(size_t j; 0 .. N.numCols)
        assert(N[i,j] == (1.0 * 2.0 * 2.0));

    // Test division
    M = Matrix.matrixOf(1.0,3,3);
    N = M / 2.0;
    foreach(size_t i; 0 .. N.numRows)
      foreach(size_t j; 0 .. N.numCols)
        assert(N[i,j] == (1.0 / 2.0));
  }

  /**
  * Overloaded multiplication/division by a scalar.
  */
  ref Matrix opOpAssign(string op)(in double rhs)
    if(op == "*" || op == "/")
  {
    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        size_t rw = r * cols;
        mixin("m[rw .. (rw + cols)] " ~ op ~"= rhs;");
      }
    }
    else
    {
      // Now do the element by element op
      mixin("m[] " ~ op ~ "= rhs;");
    }

    return this;
  }

  unittest
  {
    mixin(announceTest("opOpAssign * / double"));

    // Test multiplication
    Matrix M = Matrix.matrixOf(1.0,3,3);
    M *= 2.0;
    foreach(size_t i; 0 .. M.numRows)
      foreach(size_t j; 0 .. M.numCols)
        assert(M[i,j] == (1.0 * 2.0));
    
    // Test division
    M = Matrix.matrixOf(1.0,3,3);
    M /= 2.0;
    foreach(size_t i; 0 .. M.numRows)
      foreach(size_t j; 0 .. M.numCols)
        assert(M[i,j] == (1.0 / 2.0));
  }

  /*============================================================================
  *                           Matrix Multiplication
  *===========================================================================*/
  /// Overloaded multiplication of matrices..
  Matrix opBinary(string op)(in Matrix rhs) const
    if( op == "*")
  {
    // Check to make sure the cols of the left side == rows of right side
    assert(cols == rhs.rows,"Multiplication dimemsions mismatch.");

    // Make a new matrix
    Matrix temp = Matrix(rows,rhs.cols, 0.0);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        foreach(c; 0 .. rhs.cols)
        {
          // temp[r,c] = 0.0; initialized above
          foreach(k; 0 .. cols)
          {
            temp[r,c] += m[r * cols + k] * rhs.m[k * rhs.cols + c];
          }
        }
      }
    }
    else
    {
      foreach(r; 0 .. rows)
      {
        foreach(c; 0 .. rhs.cols)
        {
          // temp[r,c] = 0.0; initialized above
          foreach(k; 0 .. cols)
          {
            temp[r,c] += m[r * cols + k] * rhs.m[k * rhs.cols + c];
          }
        }
      }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary * Matrix"));

    double[][] m = [[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]];
    double[][] n = [[7.0, 10.0],
                    [8.0, 11.0],
                    [9.0, 12.0]];
    double[][] o = [[ 50.0,  68.0],
                   [122.0, 167.0]];
    Matrix M = Matrix(m);
    Matrix N = Matrix(n);
    Matrix O = Matrix(o);

    assert( M * N == O);
  }
  
  /**
  * Overloaded multiplication of matrices, this does the generalization of
  * the tensor product. It is used rarely, mainly with vectors, but it is used.
  */
  Matrix opBinary(string op)(in Matrix rhs) const
    if(op == "%")
  {
    // No restrictions on matrix dimensions - this can get out of hand quick...

    // Make a new matrix
    Matrix temp = Matrix(rows * rhs.rows,cols * rhs.cols);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(temp.rows)))
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   m[(r / rows) * cols + (c / cols)] * 
                   rhs.m[(r % rhs.rows) * rhs.cols + (c % rhs.cols)];
        }
      }
    }
    else
    {
      foreach(r; 0 .. temp.rows)
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   m[(r / rows) * cols + (c / cols)] * 
                   rhs.m[(r % rhs.rows) * rhs.cols + (c % rhs.cols)];
        }
      }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary % Matrix"));

    // Test multiplication
    double[][] m = [[1.0, 2.0],
                    [3.0, 4.0]];
    
    double[][] n = [[5.0, 6.0],
                   [7.0, 8.0]];
    
    double[][] o = [[ 5.0,  6.0, 10.0, 12.0],
                    [ 7.0,  8.0, 14.0, 16.0],
                    [15.0, 18.0, 20.0, 24.0],
                    [21.0, 24.0, 28.0, 32.0]];
    
    Matrix M = Matrix(m);
    Matrix N = Matrix(n);
    Matrix O = Matrix(o);
    Matrix P = M % N;

    assert(P == O);
  }

  /*============================================================================
  *                          General Methods
  *===========================================================================*/
  /**
  * Transpose a matrix. This method creates a new Matrix and fills it with the
  * transposed values.
  */
  @property Matrix T() const
  {
    Matrix temp = Matrix(cols, rows);

    version(par)
    {
      foreach(r; parallel(CountRange(rows)))
      {
        foreach(c; 0 .. cols){
          temp.m[c * rows + r] = m[r * cols + c];
        }
      }
    }
    else
    {
      for(size_t r = 0; r < rows; ++r)
      {
        for(size_t c = 0; c < cols; ++c)
        {
          temp.m[c * rows + r] = m[r * cols + c];
        }
      }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("T"));

    // Test transpose
    double[][] m = [[1.0, 2.0],
                    [3.0, 4.0]];
    double[][] n = [[1.0, 3.0],
                    [2.0, 4.0]];

    Matrix M = Matrix(m);
    Matrix N = Matrix(n);
    Matrix O = M.T;

    assert(N == O);

    // Check to make sure we didn't overwrite M
    assert(M[0,0] == 1.0);
    assert(M[0,1] == 2.0);
    assert(M[1,0] == 3.0);
    assert(M[1,1] == 4.0);

    assert(O[0,0] == 1.0);
    assert(O[0,1] == 3.0);
    assert(O[1,0] == 2.0);
    assert(O[1,1] == 4.0);
  }

  /**
  * Get a transpose view of this matrix.
  */
  @property package TransposeView Tv() const
  {
    return TransposeView(&this);
  }

  unittest
  {
    mixin(announceTest("Tv"));

    double[][] arr = [[0.0, 1.0, 2.0],
                      [0.1, 1.1, 2.1],
                      [0.2, 1.2, 2.2]];
                      
    Matrix N = Matrix(arr);
    auto tv = N.Tv;
    
    // Make sure it actually transposes
    foreach(r; 0 .. 3)
     foreach(c; 0 .. 3)
       assert(N[r,c] == tv[c,r]);
       
    // Make sure it actually transposes
    foreach(r; 0 .. 3)
     foreach(c; 0 .. 3)
       assert(N[r,c] == N.Tv[c,r]);
  }

  /**
   * Does an element-by-element comparison with std.math.approxEqual to see
   * if these matrices are approximately equal.
   */
  static bool approxEqual(in Matrix lhs, in Matrix rhs)
  {
    // Are they the same Matrix? If so, they are more than approxEqual!
    if(lhs is rhs) return true;
    
    // Then check they are the same size.
    if(lhs.rows != rhs.rows || lhs.cols != rhs.cols) return false;

    // Compare them element by element
    foreach(i; 0 .. lhs.numVals)
      if(!std.math.approxEqual(lhs.m[i],rhs.m[i])) 
        return false;

    // We haven't been able to eliminate anything! So they must be close.
    return true;
  }
}

/*==============================================================================
*                            Transpose View
*============================================================================*/
/**
* Provides a view of the matrix with transposed indicies for efficient access
* to transposed matrices without copying them. Since this stores a pointer to
* a struct, it should not be released to a scope beyond the struct. It is only
* intended to be used to read the values of a Matrix from a transposed
* perspective.
*/
package struct TransposeView
{
  // Store a reference to the source matrix.
  const Matrix *src = null;
  
  this(const Matrix *d) { src = d; }
  
  /**
  * Disable copying so this can never be returned (at least by value).
  */
  @disable this(this);
  
  /*============================================================================
  *                        Index Operators and Limits
  *==========================================================================*/
  /**
  * Get a value from the Matrix with transposed indicies, overload opIndex
  *
  * Example uses - double val = m[1,2];
  *
  *
  * Params: r - the row number
  *         c - the column number
  *
  */
  double opIndex(in size_t r, in size_t c) const
  {
    // Bounds check, this goes away with release builds
    assert(c < src.rows && r < src.cols,"Index out of bounds error.");

    return src.m[c * src.cols + r];
  }

  unittest
  {
    mixin(announceTest("opIndex"));

    double[][] arr = [[0.0, 1.0, 2.0],
                      [0.1, 1.1, 2.1],
                      [0.2, 1.2, 2.2]];
                      
    Matrix N = Matrix(arr);
    TransposeView tv = TransposeView(&N);
    
    // Make sure it actually transposes
    foreach(r; 0 .. 3)
     foreach(c; 0 .. 3)
       assert(N[r,c] == tv[c,r]);
  }
  
  /**
  * Get the number of rows in this matrix.
  */
  @property size_t numRows() { return src.cols; }

  /**
  * Get the number of columns in this matrix.
  */
  @property size_t numCols() { return src.rows; }

  unittest
  {
    mixin(announceTest("numRows numCols"));

    TransposeView M = Matrix(800,500).Tv;
    assert(M.numRows == 500);
    assert(M.numCols == 800);
  }
  
  /*============================================================================
  *                  Addition/Subtraction Operator Overloads
  *===========================================================================*/
  /**
  * Overloaded addition and subtraction operators.
  */
  Matrix opBinary(string op)(in Matrix rhs) const
    if( op == "+" || op == "-")
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(src.cols == rhs.rows && src.rows == rhs.cols);

    // Make a new matrix
    Matrix temp = Matrix(src.cols,src.rows);

    version(par)
    {
      // Now do the element by element op
      foreach(r; parallel(CountRange(src.cols)))
        foreach(c; 0 .. src.rows)
        {
          mixin("temp.m[r * temp.cols + c] = src.m[c * src.cols + r] " ~ op ~ 
                            " rhs.m[r * rhs.cols + c];");
        }
    }
    else
    {
      // Now do the element by element op
      foreach(r; 0 .. src.cols)
        foreach(c; 0 .. src.rows)
        {
          mixin("temp.m[r * temp.cols + c] = src.m[c * src.cols + r] " ~ op ~ 
                            " rhs.m[r * rhs.cols + c];");
        }
    }

    return temp;
  }

  /// ditto
  Matrix opBinaryRight(string op)(in Matrix lhs) const
    if( op == "+" || op == "-")
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(src.cols == lhs.rows && src.rows == lhs.cols);

    // Make a new matrix
    Matrix temp = Matrix(src.cols,src.rows);

    version(par)
    {
      // Now do the element by element op
      foreach(r; parallel(CountRange(src.cols)))
        foreach(c; 0 .. src.rows)
        {
          mixin("temp.m[r * temp.cols + c] = lhs.m[r * lhs.cols + c] " ~ op ~ 
                            " src.m[c * src.cols + r];");
        }
    }
    else
    {
      // Now do the element by element op
      foreach(r; 0 .. src.cols)
        foreach(c; 0 .. src.rows)
        {
          mixin("temp.m[r * temp.cols + c] = lhs.m[r * lhs.cols + c] " ~ op ~ 
                            " src.m[c * src.cols + r];");
        }
    }

    return temp;
  }
  
  /// ditto
  Matrix opBinary(string op)(in ref TransposeView rhs) const
    if( op == "+" || op == "-")
  {
    // Force them to be the same size. Note this goes away in release builds.
    assert(src.rows == rhs.src.rows && src.cols == rhs.src.cols);
    
    // Make a new matrix
    Matrix temp = Matrix(src.cols,src.rows);

    version(par)
    {
      // Now do the element by element op
      foreach(r; parallel(CountRange(src.cols)))
        foreach(c; 0 .. src.rows)
        {
             mixin("temp.m[r * temp.cols + c] = src.m[c * src.cols + r] " ~ op ~
                        " rhs.src.m[c * rhs.src.cols + r];");
        }
    }
    else
    {
      // Now do the element by element op
      foreach(r; 0 .. src.cols)
        foreach(c; 0 .. src.rows)
        {
             mixin("temp.m[r * temp.cols + c] = src.m[c * src.cols + r] " ~ op ~
                        " rhs.src.m[c * rhs.src.cols + r];");
        }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary + - Matrix TransposeView"));

    // Test addition
    double[][] m = [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]];
    double[][] n = [[1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0]];
    double[][] o = [[2.0, 5.0, 8.0],
                    [6.0, 9.0,12.0]];
                    
    Matrix M = Matrix(m);
    Matrix MT = M.T;
    TransposeView MTv = M.Tv;
    
    Matrix N = Matrix(n);
    Matrix NT = N.T;
    TransposeView NTv = N.Tv;
    
    Matrix O = Matrix(o);
    Matrix OT = O.T;
    TransposeView OTv = O.Tv;
    
    assert(M + NTv == O);
    assert(M + NT == O);
    assert(MTv + N == O.T);
    assert(MT + N == O.T);
    assert(OT - MTv == N);
    assert(OTv - MT == N);
    assert(OTv - MTv == N);
  }
  
  /*============================================================================
  *                Multiplication and Division by scalars
  *===========================================================================*/
  /**
  * Overloaded multiplication and divistion operators for operations with a
  * Matrix and scalar.
  */
  Matrix opBinary(string op)(in double rhs) const
    if( op == "*" || op == "/")
  {

    // Make a new matrix
    Matrix temp = Matrix(src.cols,src.rows);

    version(par)
    {
      // Now do the element by element op
      foreach(r; parallel(CountRange(temp.rows)))
      {
        foreach(c; 0 .. temp.cols)
        {
          mixin(
            "temp.m[r * temp.cols + c]  = src.m[c * src.cols + r] "~op~" rhs;"
          );
        }
      }
    }
    else
    {
      // Now do the element by element op
      foreach(r; 0 .. temp.rows)
      {
        foreach(c; 0 .. temp.cols)
        {
          mixin(
            "temp.m[r * temp.cols + c]  = src.m[c * src.cols + r] "~op~" rhs;"
          );
        }
      }
    }

    return temp;
  }

  /**
  * Overloaded multiplication and divistion operators for operations with a
  * Matrix and scalar. Only allow multiplication from the left, it is
  * undefined what scalar / Matrix is.
  */
  Matrix opBinaryRight(string op)(in double lhs) const
    if( op == "*")
  {
    // Commutative property of scalar multiplication
    return opBinary!op(lhs);
  }

  unittest
  {
    mixin(announceTest("opBinary * / double"));

    double[][] m = [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]];

    double[][] n = [[2.0,  8.0],
                    [4.0, 10.0],
                    [6.0, 12.0]];
              
    Matrix M = Matrix(m);
    Matrix MT = M.T;
    TransposeView MTv = M.Tv;
    
    Matrix N = Matrix(n);
    TransposeView NTv = N.Tv;

    assert(MTv * 2.0 == 2.0 * MTv);
    assert(MTv * 2.0 == 2.0 * MT);
    assert(MTv * 2.0 == N);
    assert(NTv / 2.0 == M);
  }
  
  /*============================================================================
  *                           Matrix Multiplication
  *===========================================================================*/

  /// Matrix Multiplication
  Matrix opBinary(string op)(in Matrix rhs) const
    if( op == "*")
  {
    // Check to make sure the 'rows' of the left side == rows of right side
    assert(src.rows == rhs.rows,"Multiplication dimemsions mismatch.");

    // Make a new matrix
    Matrix temp = Matrix(src.cols,rhs.cols);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(src.cols)))
      {
        foreach(c; 0 .. rhs.cols)
        {
          temp[r,c] = 0.0;
          foreach(k; 0 .. src.rows)
          {
            temp[r,c] += src.m[k * src.cols + r] * rhs.m[k * rhs.cols + c];
          }
        }
      }
    }
    else
    {
      foreach(r; 0.. src.cols)
      {
        foreach(c; 0 .. rhs.cols)
        {
          temp[r,c] = 0.0;
          foreach(k; 0 .. src.rows)
          {
            temp[r,c] += src.m[k * src.cols + r] * rhs.m[k * rhs.cols + c];
          }
        }
      }
    }

    return temp;
  }

  /// ditto
  Matrix opBinaryRight(string op)(in Matrix lhs)const
    if( op == "*")
  {
    // Check to make sure the 'rows' of the left side == rows of right side
    assert(src.cols == lhs.cols,"Multiplication dimemsions mismatch.");

    // Make a new matrix
    Matrix temp = Matrix(lhs.rows,src.rows);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(lhs.rows)))
      {
        foreach(c; 0 .. src.rows)
        {
          temp[r,c] = 0.0;
          foreach(k; 0 .. lhs.cols)
          {
            temp[r,c] += src.m[c * src.cols + k] * lhs.m[r * lhs.cols + k];
          }
        }
      }
    }
    else
    {
      foreach(r; 0 .. lhs.rows)
      {
        foreach(c; 0 .. src.rows)
        {
          temp[r,c] = 0.0;
          foreach(k; 0 .. lhs.cols)
          {
            temp[r,c] += src.m[c * src.cols + k] * lhs.m[r * lhs.cols + k];
          }
        }
      }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary * Matrix"));

    double[][] m = [[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]];

    double[][] n = [[2.0,  4.0,  6.0],
                    [8.0, 10.0, 12.0]];

    double[][] o = [[34.0, 44.0, 54.0],
                    [44.0, 58.0, 72.0],
                    [54.0, 72.0, 90.0]];

    double[][] p = [[28.0,  64.0],
                   [64.0, 154.0]];

    Matrix M = Matrix(m);
    Matrix MT = M.T;
    TransposeView MTv = M.Tv;
    
    Matrix N = Matrix(n);
    Matrix NT = N.T;
    TransposeView NTv = N.Tv;

    Matrix O = Matrix(o);

    Matrix P = Matrix(p);

    assert(MTv * N == O);
    assert(MTv * N == MT * N);

    assert(M * NTv == P);
    assert(M * NTv == M * NT);
  }
  
  /**
  * Overloaded multiplication of matrices, this does the generalization of
  * the tensor product. It is used rarely, mainly with vectors, but it is used.
  */
  Matrix opBinary(string op)(in Matrix rhs) const
    if(op == "%")
  {
    // No restrictions on matrix dimensions

    // Make a new matrix
    Matrix temp = Matrix(src.cols * rhs.rows,src.rows * rhs.cols);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(temp.rows)))
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   src.m[(c / src.rows) * src.cols + (r / src.cols)] * 
                   rhs.m[(r % rhs.rows) * rhs.cols + (c % rhs.cols)];
        }
      }
    }
    else
    {
      foreach(r; 0 .. temp.rows)
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   src.m[(c / src.rows) * src.cols + (r / src.cols)] * 
                   rhs.m[(r % rhs.rows) * rhs.cols + (c % rhs.cols)];
        }
      }
    }

    return temp;
  }

  /// ditto
  Matrix opBinaryRight(string op)(Matrix lhs) const
    if(op == "%")
  {
    // No restrictions on matrix dimensions

    // Make a new matrix
    Matrix temp = Matrix(lhs.rows * src.cols, lhs.cols * src.rows);

    // Now do the element by element op
    version(par)
    {
      foreach(r; parallel(CountRange(temp.rows)))
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   lhs.m[(r / lhs.rows) * lhs.cols + (c / lhs.cols)] *
                   src.m[(c % src.rows) * src.cols + (r % src.cols)];
        }
      }
    }
    else
    {
      foreach(r; 0 .. temp.rows)
      {
        foreach(c; 0 .. temp.cols)
        {
          temp.m[r * temp.cols + c] = 
                   lhs.m[(r / lhs.rows) * lhs.cols + (c / lhs.cols)] *
                   src.m[(c % src.rows) * src.cols + (r % src.cols)];
        }
      }
    }

    return temp;
  }

  unittest
  {
    mixin(announceTest("opBinary %"));

    double[][] m = [[1.0, 3.0],
                    [2.0, 4.0]];
    
    double[][] n = [[5.0, 6.0],
                    [7.0, 8.0]];
    
    double[][] o = [[ 5.0,  6.0, 10.0, 12.0],
                    [ 7.0,  8.0, 14.0, 16.0],
                    [15.0, 18.0, 20.0, 24.0],
                    [21.0, 24.0, 28.0, 32.0]];

    double[][] q = [[ 5.0, 10.0,  6.0, 12.0],
                    [15.0, 20.0, 18.0, 24.0],
                    [ 7.0, 14.0,  8.0, 16.0],
                    [21.0, 28.0, 24.0, 32.0]];
    
    Matrix M = Matrix(m);
    TransposeView MTv = M.Tv;

    Matrix N = Matrix(n);

    Matrix O = Matrix(o);

    Matrix P = MTv % N;

    Matrix Q = Matrix(q);

    Matrix R = N % MTv;

    assert(P == O);
    assert(R == Q);
  }
  
  /*============================================================================
  *                           Other operators
  *===========================================================================*/
  /**
  * Overloaded opEquals.
  */
  bool opEquals(in TransposeView rhs)
  {
    // Test for same object first.
    if(this is rhs || src is rhs.src) return true;

    // Now compare them
    if(src.rows == rhs.src.rows && src.cols == rhs.src.cols)
      // Compare them element by element to be sure the values are the same.
      return src.m == rhs.src.m;

    else return false;
  }
  
  /// ditto
  bool opEquals(in Matrix rhs){

    // Compare them element by element
    if(src.cols == rhs.rows && src.rows == rhs.cols)
    {
      // Compare them element by element to be sure the values are the same.
      foreach(r; 0 .. rhs.rows)
      {
        foreach(c; 0 .. rhs.cols)
        {
          if(src.m[c * src.cols + r] != rhs.m[r * rhs.cols + c]) return false;
        }
      }
      return true;
    }

    else return false;
  }

  unittest
  {
    mixin(announceTest("opEquals"));

    Matrix M = Matrix.matrixOf(3.14159, 80, 4);
    Matrix N = M;
    Matrix O = Matrix.matrixOf(1.0, 80, 4);
    Matrix P = Matrix.matrixOf(3.14159, 79, 4);

    assert(M.Tv == M.Tv);
    assert(M.Tv == N.Tv);
    assert(M.T == N.Tv);
    assert(M != M.Tv);
  }
  
  /**
  * Overloaded opUnary!"-"
  */
  Matrix opUnary(string op)() const if(op == "-") { return -src.T; }

  unittest
  {
    mixin(announceTest("opUnary"));

    double[][] m = [[1.0, 3.0],
                   [2.0, 4.0]];
    double[][] n = [[-1.0, -2.0],
                   [-3.0, -4.0]];

    Matrix M = Matrix(m);
    Matrix N = Matrix(n);
    Matrix O = -M.Tv;

    assert(N == O);

    assert(O[0,0] == -1.0);
    assert(O[0,1] == -2.0);
    assert(O[1,0] == -3.0);
    assert(O[1,1] == -4.0);
  }

  /*============================================================================
  *                     Just give me a Matrix already
  *===========================================================================*/
  @property Matrix matrix() const { return src.T; }
}

/*==============================================================================
*                       SVD decomposition of a matrix
*=============================================================================*/
/**
* SVD decomposition of a matrix.
*/
struct SVDDecomp
{
  // Work with a custom allocator potentially, use whatever matrix is using
  private IAllocator localAlloc_;

  private Matrix u;
  private size_t m;  // Rows of original Matirx
  private size_t n;  // Columns of original Matrix
  private double[] w;
  private Matrix v;
  private double cond; // Condition number of matrix from the decomp

  /**
  * Create an SVDDecomp from this matrix.
  *
  * The decomposition is performed in this constructor.
  *
  */
  this(in Matrix a)
  {
    localAlloc_ = Matrix.classAlloc;

    this.u = a.dup;      // Copy in via postblit
    m = a.rows;
    n = a.cols;

    this.w = localAlloc_.makeArray!double(n);
    this.v = Matrix.zeros(n);

    cond = double.nan;  // Calculate this later if needed.

    this.decompose();
  }

  /**
  * If passed a Transpose view, just create the new matrix and work with that. 
  */
  this(in TransposeView aT) { this(aT.matrix); }

  /**
  * Destructor required to free memory.
  */
  ~this(){ if(w) localAlloc_.dispose(w); }

  private void decompose()
  {
    double[] rv1 = localAlloc_.makeArray!double(this.n);
    scope(exit) localAlloc_.dispose(rv1);
    size_t l;

    double g = 0.0;
    double scale = 0.0;
    double anorm = 0.0;

    foreach(size_t i; 0 .. this.n)
    {
      l = i + 1;
      rv1[i] = scale * g;
      double s = 0.0;
      g = scale = 0.0;
      if(i < this.m)
      {
        for(auto k = i; k < this.m; ++k) scale += abs(this.u[k,i]);
        if(scale)
        {
          for(auto k = i; k < this.m; ++k)
          {
            this.u[k,i] /= scale;
            s += this.u[k,i] *this.u[k,i];
          }
          double f = this.u[i,i];
          g = -sqrt(s) * sgn(f);
          double h = f * g - s;
          this.u[i,i] = f - g;
          for(auto j = l; j < this.n; ++j)
          {
            s = 0.0;
            for(auto k = i; k < this.m; ++k) s += this.u[k,i] * this.u[k,j];
            f = s / h;
            for(auto k = i; k < this.m; ++k) this.u[k,j] += f * this.u[k,i]; 
          }
          for(auto k = i; k < this.m; ++k) this.u[k,i] *= scale;
        }
      }
      this.w[i] = scale * g;
      g = s = scale = 0.0;
      if(i < this.m && i != this.n)
      {
        for(auto k = l; k < this.n; ++k) scale += abs(this.u[i,k]);
        if(scale)
        {
          for(auto k = l; k < this.n; ++k)
          {
            this.u[i,k] /= scale;
            s += this.u[i,k] * this.u[i,k];
          }
          double f = this.u[i, l];
          g = -sqrt(s) * sgn(f);
          double h = f * g - s;
          this.u[i,l] = f - g;
          for(auto k = l; k < this.n; ++k) rv1[k] = this.u[i,k] / h;
          for(auto j = l; j< this.m; ++j)
          {
            s = 0.0;
            for(auto k = l; k < this.n; ++k) s += this.u[j,k] * this.u[i,k];
            for(auto k = l; k < this.n; ++k) this.u[j,k] += s * rv1[k];
          }
        for(auto k = l; k < this.n; ++k) this.u[i,k] *= scale;
        }
      }
      anorm = max(anorm,(abs(this.w[i]) + abs(rv1[i])));
    }

    for(auto i = this.n - 1; i >= 0 && i < this.n; --i)
    {
      if(i < n - 1)
      {
        if(g)
        {
          for(auto j = l; j < this.n; ++j) 
            this.v[j,i] = (this.u[i,j] / this.u[i,l]) / g;
          for(auto j = l; j < this.n; ++j)
          {
            double s = 0.0;
            for(auto k = l; k < this.n; ++k) s += this.u[i,k] * this.v[k,j];
            for(auto k = l; k < this.n; ++k) this.v[k,j] += s * this.v[k,i];
          }
        }
        for(auto j = l; j < this.n; ++j) {this.v[i,j] = 0.0; this.v[j,i] = 0.0;}
      }
      this.v[i,i] = 1.0;
      g = rv1[i];
      l = i;
    }

    for(auto i = min(this.m, this.n) - 1; i >= 0 && i < min(m, n); --i)
    {
      l = i + 1;
      g = this.w[i];
      for(auto j = l; j < this.n; ++j) this.u[i,j] = 0.0;
      if(g)
      {
        g = 1.0 / g;
        for(auto j = l; j < this.n; ++j)
        {
          double s = 0.0;
          for(auto k = l; k < this.m; ++k) s += this.u[k,i] * this.u[k,j];
          auto f = (s / this.u[i,i]) * g;
          for(auto k = i; k < this.m; ++k) this.u[k,j] += f * this.u[k,i];
        }
        for(auto j = i; j < this.m; ++j) this.u[j,i] *= g;
      }
      else { for(auto j = i; j < this.m; ++j) this.u[j,i] = 0.0; }
      this.u[i,i] += 1.0;
    }

    for(auto k = this.n - 1; k >= 0 && k < this.n; --k)
    {
      for(uint its = 0; its < 60; ++its)
      {
        bool flag = true;
        size_t nm;
        for(l = k; l >= 0 && l <= k; --l)
        {
          nm = l - 1;
          if((abs(rv1[l]) + anorm) == anorm)
          {
            flag = false;
            break;
          }
          if((abs(this.w[nm]) + anorm) == anorm) break;
        }
        if(flag)
        {
          double c = 0.0;
          double s = 1.0;
          for(auto i = l; i <= k; ++i)
          {
            auto f = s * rv1[i];
            rv1[i] *= c;
            if((abs(f) + anorm) == anorm) break;
            g = this.w[i];
            auto h = hypot(f, g);
            this.w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = -f * h;
            for(size_t j = 0; j < this.m; ++j)
            {
              double y = this.u[j, nm];
              double z = this.u[j, i];
              this.u[j, nm] = y * c + z * s;
              this.u[j, i] = z * c - y * s;
            }
          }
        }
        double z = this.w[k];
        if( l == k)
        {
          if(z < 0.0)
          {
            this.w[k] = -z;
            for(size_t j = 0; j < this.n; ++j) this.v[j,k] = -this.v[j,k];
          }
          break;
        }
        if(its == 60) assert(0, "No convergence in 60 iterations");
        double x = this.w[l];
        nm = k - 1;
        double y = this.w[nm];
        g = rv1[nm];
        double h = rv1[k];
        double f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
        g = hypot(f, 1.0);
        f = (( x - z) * (x + z) + h * ((y / (f + abs(g) * sgn(f))) - h)) / x;
        double c = 1.0; 
        double s = 1.0;
        for(auto j = l; j <= nm; ++j)
        {
          auto i = j + 1;
          g = rv1[i];
          y = this.w[i];
          h = s * g;
          g = c * g;
          z = hypot(f,h);
          rv1[j] = z;
          c = f / z;
          s = h / z;
          f = x * c + g * s;
          g = g * c - x * s;
          h = y * s;
          y *= c;
          for(size_t jj = 0; jj < this.n; ++jj)
          {
            x = this.v[jj,j];
            z = this.v[jj,i];
            this.v[jj,j] = x * c + z * s;
            this.v[jj,i] = z * c - x * s;
          }
          z = hypot(f, h);
          this.w[j] = z;
          if(z){
            z = 1.0 / z;
            c = f * z;
            s = h * z;
          }
          f = c * g + s * y;
          x = c * y - s * g;
          for(size_t jj = 0; jj < this.m; ++jj)
          {
            y = this.u[jj,j];
            z = this.u[jj,i];
            this.u[jj,j] = y * c + z * s;
            this.u[jj,i] = z * c - y * s;
          }
        }
        rv1[l] = 0.0;
        rv1[k] = f;
        this.w[k] = x;
      }
    }
  }

  unittest
  {
    mixin(announceTest("SVDDecomp - decompose, U, V, W"));

    double[][] m = [[ 1.0,  2.0,  3.0,  4.0],
                    [ 5.0,  6.0,  7.0,  8.0],
                    [ 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0]];

    Matrix M = Matrix(m);

    SVDDecomp svd = SVDDecomp(M);

    Matrix U = svd.U;
    Matrix W = svd.W;
    Matrix V = svd.V;

    assert(Matrix.approxEqual(U * W * V.T, M));
    assert(Matrix.approxEqual(U.Tv * U, V.Tv * V));
    assert(Matrix.approxEqual(U.T * U, Matrix.identity(U.numCols)));
    assert(Matrix.approxEqual(V.T * V, Matrix.identity(V.numCols)));

    M = Matrix.random(100);
    svd = SVDDecomp(M);
    U = svd.U;
    W = svd.W;
    V = svd.V;

    assert(Matrix.approxEqual(U * W * V.T, M));
    assert(Matrix.approxEqual(U.Tv * U, V.Tv * V));
    assert(Matrix.approxEqual(U.T * U, Matrix.identity(U.numCols)));
    assert(Matrix.approxEqual(V.T * V, Matrix.identity(V.numCols)));
  }

  /**
  * Returns: The condition of the matrix.
  */
  @property double condition()
  {
    if(isNaN(this.cond))
    {
      this.cond = this.w[0] / this.w[$ - 1];
    }

    return this.cond;
  }

  unittest
  {
    mixin(announceTest("SVDDecomp - condition"));

    double[][] m = [[ 1.0,  2.0,  3.0,  4.0],
                    [ 5.0,  6.0,  7.0,  8.0],
                    [ 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0]];

    Matrix M = Matrix(m);

    SVDDecomp svd = SVDDecomp(M);

    M = Matrix.random(12);
    svd = SVDDecomp(M);
  }

  @property Matrix U() { return this.u; }
  @property ref const(Matrix) U() const { return this.u; }
  @property Matrix V() { return this.v; }
  @property ref const(Matrix) V() const { return this.v; }
  @property Matrix W() const
  {
    Matrix wmat = Matrix.zeros(this.n);
    foreach(i; 0 .. this.n) wmat[i,i] = this.w[i];
    return wmat;
  }

  /**
  * Returns: The pseudo-inverse of the original matrix. If this is a square
  * matrix with all non-zero singular values, then this is the actual inverse
  * as well. 
  *
  * Very small singular values are set to zero after inverting W (of which 
  * they are very large in).
  */
  @property Matrix pseudoInverse() const
  {
    Matrix Ww = Matrix.identity(this.n);

    double eps = this.m * double.min_normal;
    for(size_t i = this.n - 1; i >= 0 && i < this.n; --i){
      if(this.w[i] / this.w[0] < eps) Ww[i,i] = 0.0;
      else Ww[i,i] = 1.0 / this.w[i];
    }

    return this.v * Ww * this.u.Tv;
  }
}

/*==============================================================================
*   Vectors - convenient way of looking at matrices with 1 row or 1 column
*=============================================================================*/
/**
* Convenience methods for creating vectors, which are really just simple
* matrices.
*/
alias Vector = Matrix;

/**
* A column vector. This is a shorthand for defining a column vector.
*/
Vector CVector(size_t rows) { return Matrix(rows,1); }
Vector CVector(in double[] vals)
{
  Vector temp = Matrix(vals.length,1);
  temp.m[] = vals[];
  return temp;
}

/**
* A row vector. This is a shorthand for defining a row vector.
*/
Vector RVector(size_t cols) { return Matrix(1,cols); }
Vector RVector(in double[] vals)
{
  Vector temp = Matrix(1,vals.length);
  temp.m[] = vals[];
  return temp;
}
unittest
{
  mixin(announceTest("Vectors"));

  Vector v = CVector([1.0, 2.0, 3.0, 4.0, 5.0]);
  Vector vt = RVector([1.0, 2.0, 3.0, 4.0, 5.0]);
  Vector p = CVector([55.0]);

  assert(v.T == vt);
  assert(vt * v == p);
}

/*==============================================================================
*                        Utilities for this file
*=============================================================================*/
/**
 * Simple, quick range to replace the slice operator 0 .. NUM in foreach
 * loops used with parallel.
 */
private struct CountRange
{

  private size_t max;
  private size_t i = 0;
  private size_t step = 1;

  this(size_t mx) { max = mx; }

  this(size_t mx, size_t stepSize)
  {
    max = mx; 
    step = stepSize;
  }

  this(size_t min, size_t mx, size_t stepSize)
  {
    i = min;
    max = mx;
    step = stepSize;
  }

  @property bool empty() { return i >= max; }

  @property size_t front() { return i; }

  void popFront() { i += step; }

  int opApply(int delegate(ref size_t) dg)
  {
    int result = 0;

    for(size_t i = 0; i != max; i+=step)
    {
      result = dg(i);

      if(result) break;
    }
    return result;
  }
}

unittest
{
  mixin(announceTest("CountRange"));
  
  // CountRange
  size_t k = 0;
  foreach(i; CountRange(4))
  {
    assert(i == k);
    ++k;
  }
  k = 0;
  foreach(i; CountRange(10,2))
  {
    assert(i == k);
    k += 2;
  }
}

/*==============================================================================
*              Run through selected functions and time the code.
*=============================================================================*/
version(prof)
{
  // TODO update to test allocators

  import std.algorithm;
  static import std.compiler;
  import std.datetime;
  import std.stdio;
  import std.string;
  import std.experimental.allocator;
  import std.experimental.allocator.building_blocks;

  void main()
  {

    // Set up a file name dependent on compiler and version options
    string prefix = std.compiler.name ~ "_";
    version(par) { prefix ~= "Parallel_"; }
    else{ prefix ~= "Serial_"; }

    enum iter = 500;                  // Number of times to do it while timing.
    auto sizes = LogRange(1, 1_000);  // Matrix sizes to use

    // Allocators to test
    IAllocator[string] allocators;
    allocators["GC"] = processAllocator;
    auto ft = FreeTree!(GCAllocator)();
    allocators["FreeTree"] = allocatorObject(&ft);

    // Temporarily store results here before saving.
    TickDuration[size_t] results;

    /*
    * Make the same basic foreach block for each test. Use this with mixin to
    * create a block of code to time a function.
    *
    * Params:
    * msg        - a description of what is being timed.
    * setup      - any code that needs to be evaluated before the loop.
    * func       - an anonymous function that will go inside the loop and be 
    *              timed.
    * fname      - the name of the file to output the results.
    * matSizeVar - an array literal listing the matrix sizes you want to 
    *              time for. If none is given, the variable sizes initialized
    *              above is used.
    * iterations - the number of times you want each function ran (for each 
    *              size) while while you are timing it. Defaults to the enum
    *              iter above.
    */
    string makeProfileBlock(string msg, string setup, 
                            string func, string fname,
                            string matSizeVar = "sizes",
                            string iterations = "iter"){
      return "
        foreach(allocator; allocators.keys)
        {
          foreach(sz; " ~ matSizeVar ~ ")
          {
            writef(\"Timing " ~ msg ~ ": %4d with %s\",sz, allocator); 
            stdout.flush();
            Matrix.classAlloc = allocators[allocator];
            " ~ setup ~ "
            results[sz] = benchmark!(" ~ func ~ ")(" ~ iterations ~ ")[0];
            writefln(\"%12.6f\", cast(double)results[sz].usecs/1000000.0);
            stdout.flush();
          }
          SaveData(results, prefix ~ allocator ~ \"_" ~ fname ~ "\");
          results.clear();
        }
      ";
    }

    /*
    * Time allocation and deallocation
    */
    mixin(makeProfileBlock(
      "allocation of matrixOf",
      "Matrix M;",
      "{M = Matrix.matrixOf(5.0, sz);}",
      "Allocation.csv"
      ));
    
    /*
    * Time addition
    */
    mixin(makeProfileBlock(
      "addition",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);
       Matrix N = Matrix.matrixOf(20.0, sz, sz);",
      "{Matrix O = M + N;}",
      "Addition.csv"
      ));

    /*
    * Time multiplication * matrix
    */
    mixin(makeProfileBlock(
      "multiplication",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);
       Matrix N = Matrix.matrixOf(20.0, sz, sz);",
      "{Matrix O = M * N;}",
      "Multiplication.csv"
      ));
  
    /*
    * Time multiplication % matrix
    */
    mixin(makeProfileBlock(
      "multiplication %%",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);
       Matrix N = Matrix.matrixOf(20.0, sz, sz);",
      "{Matrix O = M % N;}",
      "Multiplication2.csv",
      "[1,2,3,4,5,6,7,8,9,10,20,30]"
      ));

    /*
    * Time Transpose
    */
    mixin(makeProfileBlock(
      "transpose",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);",
      "{Matrix O = M.T;}",
      "Transpose.csv"
      ));

    /*
    * Time Some composite operations, to see effect of Transpose, compare with
    * results of block below to see influence of TransposeView stuct.
    */
    mixin(makeProfileBlock(
      "CompositionT1",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);
       Matrix N = Matrix.matrixOf(20.0, sz, sz);",
      "{Matrix O = M.T + N;}",
      "CompositionT1.csv"
      ));

    /*
    * Time Some composite operations, to see effect of Transpose View
    */
    mixin(makeProfileBlock(
      "CompositionT2",
      "Matrix M = Matrix.matrixOf(10.0, sz, sz);
       Matrix N = Matrix.matrixOf(20.0, sz, sz);",
      "{Matrix O = M.Tv + N;}",
      "CompositionT2.csv"
      ));
  }


  /**
  * Create a range of integers like 1,2,3,...,9,10,20,30,....90,100,200...
  */
  struct LogRange
  {

    private size_t max;
    private size_t i = 0;

    this(size_t mx) { max = mx; }

    this(size_t min, size_t mx)
    {
      max = mx; 
      i = min;
    }

    @property bool empty() { return i > max; }

    @property size_t front() { return i; }

    void popFront()
    {
      size_t step = 1;
      while(step <= i) step *= 10;
      if(step > 1) step /= 10;

      i += step;

    }
  }

  /**
  * Quickly save a csv file.
  */
  void SaveData(in TickDuration[size_t] data, in string filename)
  {
    // Open the file
    auto fl = File(filename, "w");

    // Column Headers
    fl.writefln("Sz,Time");

    //Print output.
    auto keys = sort!("a<b")(data.keys);
    foreach(k; keys)
    {
      fl.writefln("%d,%d",k,data[k].usecs);
    }
  }
}
