/**
* Library for numerics in D.
*
* The goal of this library is to be fast. So it is based on structs. There is
* also a compile option for building the library with parallel execution of 
* many methods v.s. regular serial execution.
*
* It is not intended to be a complete library, or anything close. I will add
* the features and functions that I need as I need them.
*
* version par - parallel execution instead of serial execution of some 
*               methods. Serial execution is the default if no version is 
*               specified.
*
* The profile version is handy for comparing potential versions and compilers
* on any specific system.
*
* This module contains several utility methods to be used across the package,
* including some exceptions.
*
* Author: Ryan Leach
* Version: 1.0.0
* Date: January 15, 2015
*/
module numeric;

import std.exception;

version(unittest)
{
  // Make these always available for unittests
  public import std.stdio: writeln, write, stdout, stderr, writefln, writef;
  public import std.string: format;
  public import std.exception;

  // Generate string for mixin that announces this test.
  string announceTest(in string msg)
  {
    return "
    write(format(\"Testing %s - %5d: %s...\",__MODULE__,__LINE__,\"" ~ msg ~"\"));
    stdout.flush();
    stderr.flush();
    scope(exit)
    {
      writeln(\"done.\");
      stdout.flush();
      stderr.flush();
    }";
  }
}

/**
* Signal the failure of an algorithm to converge.
*/
public class FailureToConverge: Exception
{
  /// Message about cause of exception.
  public string msg;
  /// Number of iterations algorithm took.
  public size_t iterations;
  /// Tolerance that was not reached.
  public double tolerance;
  /// Best set of parameters reached so far
  public double[] bestParms;
  /// Lowest function value achieved so far
  public double minSoFar;

  public this(string message = "", size_t its = 0, double tol = double.nan,
    double[] bParms = null, double minVal = double.nan){
    super(msg);
    this.msg = message;
    this.iterations = its;
    this.tolerance = tol;
    this.bestParms = bParms;
    this.minSoFar = minVal;
  }
}

/**
* Signal the failure of an algorithm due to encountering NaN.
*/
public class NaNException: Exception
{
  /// Message about cause of exception.
  public string msg;

  public this(string message = ""){
    super(msg);
    this.msg = message;
  }
}

/**
* Signal the failure of an algorithm due to encountering Infinity.
*/
public class InfinityException: Exception
{
  /// Message about cause of exception.
  public string msg;

  public this(string message = "")
  {
    super(msg);
    this.msg = message;
  }
}
