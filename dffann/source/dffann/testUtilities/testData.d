/**
 * Author: Ryan Leach
 * Version: 1.0.0
 * Date: June6, 2015
 *
 * This module contains utilities for automatically generating test data or
 * example data, mainly for use in testing (not necessarily unittesting).
 *
 */
module dffann.testUtilities.testData;

import std.algorithm.iteration;
import std.algorithm.sorting;
import std.array;
import std.conv;
import std.file;
import std.math;
import std.stream;
import std.random;

import numeric.random: gasdev;

/**
 * Custom Exception for the test suite. If you're doing anything other than a 
 * unitttest this might be useful.
 */
public class TestUtilityException: Exception
{
  this(string message){ super(message); }
}

/**
 * Make an array filled with random numbers.
 *
 * Params:
 * numPoints   = The number of rows in the array, simulating the number of 
 *               points in a training data set.
 * binaryFlags = An array whose values correspond to the columns of the 
 *               array. If a value is true, then the random values in that
 *               column will always be zero or 1.
 *
 * Returns: An array filled with random values.
 */
public double[][] makeRandomArray(in size_t numPoints, in bool[] binaryFlags)
{
  // The number of columns
  immutable size_t numVals = binaryFlags.length;

  double[][] toRet = new double[][](numPoints, numVals);

  foreach(i; 0 .. numPoints)
  {
    foreach(j; 0 .. numVals)
    {
      if( binaryFlags[j])
      {
        toRet[i][j] = gasdev() > 0.0 ? 1.0 : 0.0;
      }
      else
      {
        toRet[i][j] = gasdev();
      }
    }
  }

  return toRet;
}

/**
 * Make a test data set for training a linear network with a linear 
 * relationship.
 *
 * Params:
 * numPoints  = The number of data points to make, or the number of rows.
 * numInputs  = The first numInputs columns will represent independent values.
 * numOutputs = The last numOutputs columns are the dependent values calculated
 *              from the inputs.
 * noiseStd   = Each computed output will be multiplied by 1.0 plus a random
 *              number. This value is the standard deviation of that gaussian
 *              noise. So to add 5 percent error, this value should be 0.05.
 *
 * Returns: An array of data that can be used to train a linear network. The
 *          values will not be normalized. 
 */
public double[][] makeLinearRegressionData(in size_t numPoints, 
                                           in size_t numInputs, 
                                           in size_t numOutputs, 
                                           in double noiseStd)
{
  // The number of columns
  immutable size_t numVals = numInputs + numOutputs;

  double[][] toRet = new double[][](numPoints, numVals);

  // Make a shift and a scale to apply to random numbers for each input column
  // This is to give each column a different distribution
  double[] inShift = new double[numInputs];
  double[] inScale = new double[numOutputs];

  foreach(i; 0 .. numInputs)
  {
    inShift[i] = gasdev() * 100.0;
    inScale[i] = uniform(0.2, 10.0);
  }

  // Build the inputs and outputs
  foreach(p; 0 .. numPoints)
  {
    // Inputs
    foreach(i; 0 .. numInputs)
    {
      toRet[p][i] = gasdev() * inScale[i] + inShift[i];
    }

    // Outputs
    foreach(o; numInputs .. numVals)
    {
      //Add a bias depending on the index
      toRet[p][o] = o;

      // Add a linear combination of the inputs
      foreach(i; 0 .. numInputs)
      {
        toRet[p][o] += sqrt(i + 1.0) * toRet[p][i];
      }

      // Add the noise factor
      toRet[p][o] *= (1.0 + gasdev() * noiseStd);

    }
  }

  return toRet;
}

/**
 * Make a test data set for training a regression network with a nonlinear 
 * relationship.
 *
 * Params:
 * numPoints  = The number of data points to make, or the number of rows.
 * numInputs  = The first numInputs columns will represent independent values.
 * numOutputs = The last numOutputs columns are the dependent values calculated
 *              from the inputs.
 * noiseStd   = Each computed output will be multiplied by 1.0 plus a random
 *              number. This value is the standard deviation of that gaussian
 *              noise. So to add 5 percent error, this value should be 0.05.
 *
 * Returns: An array of data that can be used to train a nonlinear network. The
 *          values will not be normalized. 
 */
public double[][] makeNonlinearRegressionData(in size_t numPoints,
                                              in size_t numInputs,
                                              in size_t numOutputs,
                                              in double noiseStd)
{
  // The number of columns
  immutable size_t numVals = numInputs + numOutputs;

  double[][] toRet = new double[][](numPoints, numVals);

  // Make a shift and a scale to apply to random numbers for each input column
  // This is to give each column a different distribution
  double[] inShift = new double[numInputs];
  double[] inScale = new double[numOutputs];

  foreach(i; 0 .. numInputs)
  {
    inShift[i] = gasdev() * 100.0;
    inScale[i] = uniform(0.2, 10.0);
  }

  // Build the inputs and outputs
  foreach(p; 0 .. numPoints)
  {
    // Inputs
    foreach(i; 0 .. numInputs)
    {
      toRet[p][i] = gasdev() * inScale[i] + inShift[i];
    }

    // Outputs
    foreach(o; numInputs .. numVals)
    {
      //Add a bias depending on the index
      toRet[p][o] = o;

      // Add a nonlinear combination of the inputs
      foreach(i; 0 .. numInputs)
      {
        toRet[p][o] += sqrt(i + 1.0) * toRet[p][i] + toRet[p][i] * log(toRet[p][i]) * cos(toRet[p][i]);
      }

      // Add the noise factor
      toRet[p][o] *= (1.0 + gasdev() * noiseStd);

    }
  }

  return toRet;
}

/**
 * Make a file with random values seperated by commas.
 *
 * Params:
 * numPoints   = The number of points to put in the file (lines in the file)
 * binaryFlags = An array whose values correspond to the columns of the 
 *               array. If a value is true, then the random values in that
 *               column will always be zero or 1.
 * fileName    = The name of the file to create.
 */
public void makeRandomCSVFile(in size_t numPoints, in bool[] binaryFlags, 
  in string fileName)
{
  double[][] testData = makeRandomArray(numPoints, binaryFlags);

  Stream f = new BufferedFile(fileName, FileMode.OutNew);
  scope(exit) f.close();

  foreach(vals; testData)
  {
    string[] tmp = array(map!(to!string)(vals));
    f.writefln(join(tmp,","));
  }

}

/**
 * Make a test data set for training a classification network with a linear 
 * relationship.
 *
 * Params:
 * numPoints  = The number of data points to make, or the number of rows.
 * numInputs  = The first numInputs columns will represent independent values.
 * numOutputs = The last numOutputs columns are the dependent values calculated
 *              from the inputs.
 * noiseStd   = Each computed output will be multiplied by 1.0 plus a random
 *              number. This value is the standard deviation of that gaussian
 *              noise. So to add 5 percent error, this value should be 0.05.
 *
 * Returns: An array of data that can be used to train a linear network. The
 *          values will not be normalized. 
 */
public double[][] makeLinearClassificationnData(in size_t numPoints,
                                                in size_t numInputs,
                                                in size_t numOutputs,
                                                in double noiseStd)
{
  // The number of columns
  immutable size_t numVals = numInputs + numOutputs;

  double[][] toRet = new double[][](numPoints, numVals);
  double[] sums = new double[](numPoints);
  double[] sortedSums = new double[](numPoints);


  // Build the inputs and sums
  foreach(p; 0 .. numPoints)
  {
    // Inputs
    sums[p] = 0.0;
    foreach(i; 0 .. numInputs)
    {
      toRet[p][i] = uniform!("[]",int,int)(0,1);
      sums[p] += toRet[p][i];
    }
    sums[p] *= (1.0 + gasdev() * noiseStd);
  }

  // Sort data and find percentiles
  sortedSums = sums.dup;
  sort(sortedSums);
  size_t sizeOfPercentile = numPoints / numOutputs;
  size_t numBreaks = numOutputs - 1;
  double[] breaks = new double[](numBreaks);
  foreach(i; 0 .. numBreaks)
  {
    breaks[i] = sortedSums[(i + 1) * sizeOfPercentile];
  }

  // Outputs - classify each point by the percentile that it falls into
  foreach(p; 0 .. numPoints)
  {
    // Initialize to no class selected
    toRet[p][numInputs .. numVals] = 0.0;

    // Choose the class
    bool found = false;
    foreach(o; numInputs .. (numVals - 1))
    {

      if(sums[p] < breaks[o - numInputs])
      {
        toRet[p][o] = 1.0;
        found = true;
        break;
      }
    }
    if(!found) toRet[p][$ - 1] = 1.0;
  }

  return toRet;
}
