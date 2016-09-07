/**
* Library for numerics in D.
*
* This module contains routines for random numbers. Note that many of these
* routines were developed for floating point arithmetic, and I'm not sure how
* well they stand up at double precisiion in a 64 bit envirionment.
*
* Probably better off using the std.random library's uniform generator,
* however, after inspection of the code in std/random.d, it appears that the
* default generator is much more complicated and defaults to a 32 bit 
* implementation. It seems possible to create a 64 bit version, but will
* require some research into how exactly to use the myriad types. So for now
* will stick with these routines for random numbers.
*
* // TODO - add code to use std.random for uniform deviates, and get check for
* //        for 64 bit compilation, then configure generators correctly.
*
* Version: 1.0.0
* Date: January 25, 2015
*/
module numeric.random;

import numeric.numeric;

import std.math;
import std.random;

/**
* Routine ran0 ported from Numerical Recipes in C, Press et al, 2nd ed, 1999,
* page 279.
*
* Params: a seed that must not be altered between succesive calls.
*
* Returns: a uniform random deviate.
*/
double ran0(ref long idum){

  enum IA = 16807;
  enum IM = 2147483647;
  enum AM = 1.0 / IM;
  enum IQ = 127773;
  enum IR = 2836;
  enum MASK = 123459876;

  long k;
  double ans;

  idum ^= MASK;
  k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;
  if(idum < 0) idum += IM;
  ans = AM * idum;
  idum ^= MASK;
  return ans;
 }

/**
* Routine ran1 ported from Numerical Recipes in C, Press et al, 2nd ed, 1999,
* page 280.
*
* Params: a negative integer seed that must not be altered between succesive 
*         calls.
*
* Returns: a uniform random deviate between 0.0 and 1.0 exclusive.
*/
double ran1(ref long idum){

  enum IA = 16807;
  enum IM = 2147483647;
  enum AM = 1.0 / IM;
  enum IQ = 127773;
  enum IR = 2836;
  enum NTAB = 32;
  enum NDIV = (1 + (IM -1) / NTAB);
  enum EPS = 1.2e-7;
  enum RNMX = (1.0 - EPS);
  enum MASK = 123459876;

  int j;
  long k;
  static long iy = 0;
  static long[NTAB] iv;
  double temp;

  if( idum <= 0 || !iy){ // Initialize
    assert(idum < 0, "Argument MUST be negative on first call!");
    if(-idum < 1) idum = 1;
    else idum = - idum;
    for(j = NTAB + 7; j >= 0; --j){ // Load the shuffle table after 8 warm ups
      k = idum / IQ;
      idum = IA * (idum - k * IQ) - IR * k;
      if(idum < 0) idum += IM;
      if(j < NTAB) iv[j] = idum;
    }
    iy = iv[0];
  }
  k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;
  if(idum < 0) idum += IM;
  j = cast(int) (iy / NDIV);
  iy = iv[j];
  iv[j] = idum;
  if((temp = AM * iy) > RNMX) return RNMX;
  else return temp;
}

// TODO add ran2 and ran3

/**
* Routine gasdev ported from Numerical Recipes in C, Press et al, 2nd ed, 1999,
* page 289.
*
* Params: a negative integer seed that must not be altered between succesive 
*         calls.
*
* Returns: a gaussian random deviate of zero mean and unit standard deviation.
*/
double gasdev(ref long idum)
{
  static bool iset = false;
  static double gset;
  double fac, rsq, v1, v2;

  if(idum < 0) iset = false;
  if(!iset){
    do {
      v1 = 2.0 * ran1(idum) - 1.0;
      v2 = 2.0 * ran1(idum) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while(rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = true;
    return v2 * fac;
  }
  else{
    iset = false;
    return gset;
  }
}

/**
* Routine gasdev ported from Numerical Recipes in C, Press et al, 2nd ed, 1999,
* page 289. This override relies on the D phobos standard library random
* number generator to generate the uniform random deviates.
*
* Params: a negative integer seed that must not be altered between succesive 
*         calls.
*
* Returns: a gaussian random deviate of zero mean and unit standard deviation.
*/
double gasdev()
{
  static bool iset = false;
  static double gset;
  double fac, rsq, v1, v2;

  // Shorthand for a new uniform random deviate in an open range
  alias ranNum = uniform!("()",double,double);
  
  if(!iset){
    do {
      v1 = 2.0 * ranNum(0.0, 1.0) - 1.0;
      v2 = 2.0 * ranNum(0.0, 1.0) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while(rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = true;
    return v2 * fac;
  }
  else{
    iset = false;
    return gset;
  }
}
 