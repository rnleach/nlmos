/**
 * Library for numerics in D.
 *
 * This module contains routines for minimizing functions as defined in func.d.
 *
 * Version: 1.0.0
 * Date: January 23, 2015
 */
module numeric.minimize;

public import numeric.numeric;

import numeric.func;
import numeric.matrix;

import std.algorithm;
import std.math;
import std.string;

version(unittest)
{
  import std.stdio;
}

/**
 * Shift a point.
 *
 * Given a starting point, a direction, and a scale, return a point shifted
 * by scale times direction from starting point.
 * 
 * Used internally for minimizations of multivariable functions along a line.
 * 
 * Returns: start + scale * dir
 */
private double[] delta(double[] start, double[] dir, double scale)
{
  assert(start.length == dir.length);

  double[] toRet = start.dup;

  toRet[] += dir[] * scale;
  
  return toRet;
}
///
unittest
{
  // delta

  double[] x = [1.0, 2.0, 3.0, 4.0];
  double[] dir = [1.0, 2.0, 3.0, 4.0];
  const double scale = 0.1;

  assert(delta(x, dir, scale) == [1.1, 2.2, 3.3, 4.4]);
}

/**
 * Convenient and more expressive way to represent results of a bracketing 
 * operation compared to an array.
 *
 * If there was a failure to bracket, then the minimum value of the function
 * found during the proces should be returned via parameter ax.
 */
public struct BracketResults
{
  /// Abisca points bracketing a minimum and function values at those points.
  public double ax, bx, cx;
  /// ditto
  public double fa, fb, fc;

  /**
   * Params:
   * a = abisca value bracketing a minimum from the low side
   * b = abisca value between the brackets, with function value lower than a and c
   * c = abisca value bracketing a minimum from the high side
   * fa = function value at a
   * fb = function value at b
   * fc = function value at c
   */
  public this(double a, double b, double c, double fax, double fbx, double fcx)
  {
    ax = a;   bx = b;   cx = c; 
    fa = fax; fb = fbx; fc = fcx;
  }

  /**
   * Returns: false if any members are NaN or infinity, which is the signal for 
   *          a failed attempt to bracket.
   */
  public @property bool bracketFound() const
  {
    if(isNaN(ax) || isNaN(bx) || isNaN(cx) || 
       isNaN(fa) || isNaN(fb) || isNaN(fc)) return false;

    if(isInfinity(ax) || isInfinity(bx) || isInfinity(cx) || 
       isInfinity(fa) || isInfinity(fb) || isInfinity(fc)) return false;

    return true;
  }

  /// Create a string of these values.
  public string toString() const
  {
    return format("BracketResults[found=%s, ax=%f, bx=%f, cx=%f, fa=%f, fb=%f, fc=%f]",
                  this.bracketFound, ax, bx, cx, fa, fb, fc);
  }
}

/**
 * Bracket a minimum along a given direction.
 *
 * For doing a line minimization given a starting point and a direction,
 * return 3 values to bracket a minimum along the line through the starting 
 * point in the direction given.
 * 
 * Based on routine mnbrak from Numerical Recipes in C, Second Edition, 
 * section 10.1, pg 400. However, I've added a feature.
 *
 * If a bracket cannot be found, this routine will return the location of the 
 * lowest function value found so far in the 'ax' field of the returned struct.
 * 
 * Params:
 * startingPoint = the starting point.
 * direction     = the direction to search along a line.
 * f             = the function to evaluate when bracketing a minimum.
 *
 * Returns: a BracketResults struct with the minimum along the line bracketed
 *          by the ax and ac members. The middle location bx is included along
 *          the function values at all of those locations. If the routine was
 *          unable to bracket a minimum, all the valuse of the return will be
 *          double.nan and this can be checked for with the bracketFound method.
 */
public BracketResults bracketMinimum(double[] startingPoint, 
  double[] direction, Func f)
{

  assert(startingPoint.length == direction.length);
  
  enum GOLD = 1.618034;
  enum GLIMIT = 100.0;
  enum TINY = 1.0e-20;
  
  double ax = 0.0;
  f.evaluate(delta(startingPoint, direction, ax));
  double fa = f.value;
  
  double bx = 0.1;
  f.evaluate(delta(startingPoint, direction, bx));
  double fb = f.value;
  
  // Check for NaN
  while(isNaN(fb) || isInfinity(fb)){
    bx /= 10.0;
    if(abs(bx) < TINY) {
      return BracketResults();
    }
    f.evaluate(delta(startingPoint, direction, bx));
    fb = f.value;
  }
  
  if(fb > fa){ // Swap so fa to fb is downhill direction
    swap(fa, fb);
    swap(ax,bx);
  }
  
  double cx = bx + GOLD * (bx - ax);
  f.evaluate(delta(startingPoint, direction, cx));
  double fc = f.value;
  
  // Check for NaN
  double scale = GOLD * (bx - ax);
  while(isNaN(fc))
  {
    if(abs(scale) < TINY) {
      cx = bx < ax ? bx : ax;
      return BracketResults(cx, double.nan, double.nan, double.nan, 
        double.nan, double.nan);
    }
    scale /= 10.0;
    cx = bx + scale;
    f.evaluate(delta(startingPoint, direction, cx));
    fc = f.value;
  }
  
  while(fb > fc){
    const double r = (bx - ax) * (fb - fc);
    const double q = (bx - cx) * (fb - fa);
    double u = bx - ((bx - cx) * q - (bx - ax) * r) / 
              (2.0 * ((max(abs(q - r), TINY) >= 0.0) ? 
                (abs(q - r)) : (-abs(q - r))));

    double fu;
    const double ulim = bx + GLIMIT * (cx - bx);
    if((bx - u) * (u - cx) > 0.0){
      f.evaluate(delta(startingPoint, direction, u));
      fu = f.value;
      if(fu < fc){
        ax = bx; 
        fa = fb;
        bx = u; 
        fb = fu;
        return BracketResults(ax, bx, cx, fa, fb, fc);
      }
      else if(fu > fb){
        cx = u; 
        fc = fu;
        return BracketResults(ax, bx, cx, fa, fb, fc);
      }
      u = cx + GOLD * (cx - bx);
      f.evaluate(delta(startingPoint, direction, u));
      fu = f.value;
    }
    else if((cx - u) * (u - ulim) > 0.0){
      f.evaluate(delta(startingPoint, direction, u));
      fu = f.value;
      if(fu < fc){
        bx = cx; 
        cx = u; u = cx + GOLD *(cx - bx);
        f.evaluate(delta(startingPoint, direction, u));
        fb = fc; 
        fc = fu;  
        fu = f.value;
      }
    }
    else if((u - ulim) * (ulim - cx) >= 0.0){
      u = ulim;
      f.evaluate(delta(startingPoint, direction, u));
      fu = f.value();
    }
    else{
      u = cx + GOLD *(cx - bx);
      f.evaluate(delta(startingPoint, direction, u));
      fu = f.value;
    }
    ax = bx;
    bx = cx; 
    cx = u;
    fa = fb; 
    fb = fc; 
    fc = fu;
  }

  // Maybe fc is NaN, make sure smallest value is in ax, fa pair
  if((isNaN(fc) || isInfinity(fc)) && fa > fb)
  {
    swap(ax, bx);
    swap(fa, fb);
  }
  
  return BracketResults(ax, bx, cx, fa, fb, fc);
}

///
unittest
{
  // bracketMinimum

  /*
   * Imported function in the unittest version of func.d with an absolute 
   * minimum at (x=1, y=2, z=3).
   */
  AnotherFunction af = new AnotherFunction;

  // Start at point (1,1,1) and bracket a minimum in along a given direction
  double[] startPoint = [1.0, 1.0, 1.0];

  /* 
   * For direction (0,1,0) should bracket minimum point at (1,2,1). Since we
   * are only moving along the y direction and want to bracket the point y=2,
   * startPoint[1] + (ax,bx,cx) should bracket 2.
   */
  BracketResults results = bracketMinimum(startPoint, [0.0,1.0,0.0], af);
  assert(results.bracketFound); 
  assert(results.fa >= results.fb && results.fc >= results.fb);
  assert((results.ax > results.bx && results.bx > results.cx) ||
    (results.cx > results.bx && results.bx > results.ax));

  /* 
   * For direction (1,0,0) should bracket minimum point at (1,1,1). Since we
   * are only moving along the x direction and want to bracket the point x=1,
   * startPoint[0] + (ax,bx,cx) should bracket 1.
   */
  results = bracketMinimum(startPoint, [1.0,0.0,0.0], af);
  assert(results.bracketFound); 
  assert(results.fa >= results.fb && results.fc >= results.fb);
  assert((results.ax > results.bx && results.bx > results.cx) ||
    (results.cx > results.bx && results.bx > results.ax));

  /* 
   * For direction (0,0.5,1) should bracket minimum point at (1,2,3).
   */
  results = bracketMinimum(startPoint, [0.0,0.5,1.0], af);
  assert(results.bracketFound); 
  assert(results.fa >= results.fb && results.fc >= results.fb);
  assert((results.ax > results.bx && results.bx > results.cx) ||
    (results.cx > results.bx && results.bx > results.ax));

  /* 
   * This function has no local minimum, so it should not be able to bracket
   *  one!
   */
  ANegativeFunction nf = new ANegativeFunction;
  results = bracketMinimum(startPoint, [0.0,1.0,0.0], nf);
  assert(!results.bracketFound);
}

/**
 * Convenient and more expressive way to represent results of a line 
 * minimization operation compared to an array.
 *
 * Assuming the line minimization was done with a starting point sp and 
 * direction dr, the coordinates of the minimum will be delta(sp, dir, alpha),
 * where alpha is a member of this struct. The value at that position is
 * the value member, and the gradient is the gradient member. The gradient
 * member may be null if the gradient was not requested during minimization.
 */
public struct LineMinimizationResults
{
  /**
   * Assuming the line minimization was done with a starting point sp and
   * direction dr, alpah is the distance along that direction to get to the 
   * minimum.
   */
  public double alpha;
  /// The value of the function at the minimum.
  public double value;
  /**
   * The value of the gradient at the minimum, may be null if no gradient was 
   * calculated.
   */
  public double[] gradient;
  /// The coordinates of the minimum after the line minimization.
  public double[] pos;
  /// The number of iterations it took to get there.
  public uint iterations;
  
  /**
   * Params:
   * a  = Assuming the line minimization was done with a starting point sp and
   *      direction dr, a is the distance along that direction to get to the
   *      minimum.
   * v  = the value of the function at the minimum.
   * g  = the gradient of the function at the minimum, and may be null.
   * sp = the starting point used to do the line minimization.
   * dr = the direction used in the line minimization.
   * it = the number of iterations it took to reach the minimum.
   */
  public this(double a, double v, double[] g, double[] sp, double[] dr, uint it)
  {
    this.alpha = a;
    this.value = v;
    this.gradient = g;
    this.pos = delta(sp, dr, a);
    this.iterations = it;
  }

  public string toString() const 
  {
    return format("\nLineMinimizationResults:\niterations = %d, alpha = %f,\n" ~
      "value = %f,\ngradient = %s,\npos = %s\n", iterations, alpha, value, 
      gradient, pos);
  }
}

/**
 * Line minimization based off of routine brent from Numerical Recipes in 
 * C, Second Edition, section 10.2, pg 404.
 * 
 * Params:
 * startingPoint = the starting point
 * direction     = the direction to search along a line
 * brackets      = as returned from bracketMinimum
 * f             = the function to evaluate when minimizing.
 * 
 * Returns: an object with public access to the results.
 */
public LineMinimizationResults lineMinimize(double[] startingPoint, 
                                            double[] direction, 
                                            BracketResults brackets, 
                                            Func f)
{

  assert(brackets.bracketFound);
  
  enum ITMAX = 100; // Maximum number of iterations.
  enum CGOLD = 0.3819660;
  enum ZEPS = 1.0e-10;
  enum TOL = 1.0e-2;

  double e = 0.0;
  
  // unpack the brackets
  const double ax = brackets.ax;
  //double fa = brackets.fa;
  const double bx = brackets.bx;
  const double fb = brackets.fb;
  const double cx = brackets.cx;
  //double fc = brackets.fc;
  
  double a = (ax < cx ? ax : cx);
  double b = (ax > cx ? ax : cx);
  
  double x, fx, w, fw, v, fv;
  double[] gx;// gw, gv;
  x  = w  = v  = bx;
  fx = fw = fv = fb;
  gx = null;
  //gv = 
  //gw = null;
  
  double xmin, fxmin;
  double[] gmin, gu;
  double tol1, tol2, xm, r, q, p, etemp, d = 0.0, u, fu;
  
  uint iter = 0;
  for(; iter < ITMAX; ++iter){
    xm = 0.5 * (a + b);
    
    tol1 = TOL * abs(x) + ZEPS;
    tol2 = 2.0 * tol1;
    
    if (abs(x - xm) <= (tol2 -0.5 * (b - a))){
      xmin = x;
      // Possible gx has not been evaluatd, so re-calculating all of these
      if(!gx){
        f.evaluate(delta(startingPoint, direction, xmin), true);
        fx = f.value;
        gx = f.gradient;
      }
      fxmin = fx; 
      gmin = gx; 
      return LineMinimizationResults(xmin, fxmin, gmin, startingPoint, 
        direction, iter);
    }
    if(abs(e) > tol1 ){
      r = (x - w) * (fx -fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) *r;
      q = 2.0 * (q - r);
      if( q > 0.0) p = -p;
      q = abs(q);
      etemp = e;
      e = d;
      if(abs(p) >= abs(0.5 * q * etemp) || p <= q * (a-x) || p >= q * (b - x)){
        e = (x >= xm ? a - x: b - x);
        d = CGOLD * e;
      }
      else{
        d = p/q;
        u = x + d;
        if(u -a < tol2 || b - u < tol2)
          d = ((xm - x) >= 0.0 ? abs(tol1) : -abs(tol1));
      }
    }
    else{
      e = (x > xm ? a -x: b - x);
      d = CGOLD * e;
    }
    u = abs(d) > tol1 ? x + d: x +  (d >= 0.0 ? abs(tol1) : -abs(tol1));
    f.evaluate(delta(startingPoint,direction,u), true);
    fu = f.value;
    gu = f.gradient;
    if(fu <= fx){
      if(u >= x) a = x; else b = x;
      v = w; w = x; x = u;
      fv = fw; fw = fx; fx = fu;
      //gv = gw; 
      //gw = gx; 
      gx = gu;
    }
    else{
      if(u < x) a = u; else b = u;
      if(fu <= fw || w == x){
        v = w; fv = fw; //gv = gw;
        w = u; fw = fu; //gw = gu;
      }
      else if(fu <= fv || v == x || v== w){
        v = u; fv = fu; //gv = gu;
      }
    }
  }
  
  xmin = x;
  // Possible gx has not been evaluatd, so re-calculating all of these
  if(!gx){
    f.evaluate(delta(startingPoint, direction, xmin), true);
    fx = f.value;
    gx = f.gradient;
  }
  fxmin = fx; 
  gmin = gx; 
  return LineMinimizationResults(xmin, fxmin, gmin, startingPoint, 
    direction, iter);
}

unittest
{
  // lineMinimize

  /*
   * Imported function in the unittest version of func.d with an absolute 
   * minimum at (x=1, y=2, z=3).
   */
  AnotherFunction af = new AnotherFunction;

  // Start at point (1,1,1) and find a minimum in along a given direction
  double[] startPoint = [1.0, 1.0, 1.0];

  /* 
   * For direction (0,1,0) should find minimum point at (1,2,1), this is not 
   * a local minimum for the function, but here we are constrained to move along
   * a given line. Since we are only moving along the y direction this is at
   * (1,2,1)
   */
  double[] dr = [0.0,1.0,0.0];
  BracketResults bres = bracketMinimum(startPoint, dr, af);
  assert(bres.bracketFound);

  LineMinimizationResults lres = lineMinimize(startPoint, dr, bres, af);
  assert(approxEqual(lres.pos, [1.0, 2.0, 1.0]));
  assert(approxEqual(lres.value, 4.0));
  assert(approxEqual(lres.gradient, [0.0, 0.0, -4.0]));

  /* 
   * For direction (1,0,0) should find minimum point at (1,1,1), this is not 
   * a local minimum for the function, but here we are constrained to move along
   * a given line. Since we are only moving along the x direction this is at
   * (1,1,1).
   */
  dr = [1.0, 0.0, 0.0];
  bres = bracketMinimum(startPoint, [1.0,0.0,0.0], af);
  assert(bres.bracketFound);
  lres = lineMinimize(startPoint, dr, bres, af);
  assert(approxEqual(lres.pos, [1.0, 1.0, 1.0]));
  assert(approxEqual(lres.value, 5.0));
  assert(approxEqual(lres.gradient, [0.0, -2.0, -4.0]));

  /* 
   * For direction (0,0.5,1) should find minimum point at (1,2,3), which 
   * happens to be a local minimum of the function.
   */
  dr = [0.0, 0.5, 1.0];
  bres = bracketMinimum(startPoint, dr, af);
  assert(bres.bracketFound);

  lres = lineMinimize(startPoint, dr, bres, af);
  assert(approxEqual(lres.pos, [1.0, 2.0, 3.0]));
  assert(approxEqual(lres.value, 0.0));
  assert(approxEqual(lres.gradient, [0.0, 0.0, 0.0]));
}

/**
 * Use a BFGS algorithm to minimize the supplied function.
 * 
 * Params:
 * f         = the function to be minimized.
 * startPos  = seed for the starting position to start looking for a minimum,
 *             and the result is returned.
 * maxIt     = the maximum number of steps to take before quitting.
 * minDeltaV = a stopping condition. When the average change in value over the 
 *             last several iterations is smaller than this, stop!
 *
 */
public void bfgsMinimize(Func f, ref double[] startPos, size_t maxIt, 
  double minDeltaV)
{
  
  // These variables used to remember last XX iterations and average error to 
  // test stopping conditions.
  enum XX = 5;
  double[XX] deltaV = double.max / XX;
  
  double avgDeltaV = double.max;
  double oldV = 0.0;
  size_t idx = 0;
  
  // Initialize everything
  double[] wA = startPos;
  f.evaluate(wA, true);
  double val = f.value;
  double[] gA = f.gradient;
  
  Matrix g = CVector(gA);
  Matrix G = Matrix.identity(wA.length);
  
  // Use a break statement in one of the many stopping criteria checks.
  size_t iter = 0;
  while(true){
    
    ++iter;
    
    // Copy variables so new ones can be calculated.
    const Matrix oldg = g;
    
    // Calculate the direction to search
    Matrix d = G * g;
    
    // Get the starting point weights and direction as arrays
    double[] dA = d.m;
    
    // Bracket the minimum
    BracketResults brackets = bracketMinimum(wA, dA, f);
    
    // Failure to bracket indicates already at minimum, or none exists, so quit!
    const bool foundBracket = brackets.bracketFound;
    if(!foundBracket) {
      // If the gradient is too small, assume we failed to bracket because we
      // are at a minimum.
      if(approxEqual(gA, 0.0)) break;

      // Bracket tries to return the lowest value found so far, if it can't
      // then give up.
      if(isNaN(brackets.ax))
      {
        // Throw the best we've got so far in there
        startPos = wA;
        
        throw new FailureToConverge(
          format("Bracket Failure: %d of max %d.  " ~ 
               "Bracket Results = %s", iter, maxIt, brackets.toString()),
          iter, avgDeltaV, wA, val);
      }

      // I tried to drop the minimum value in brackets.ax, 
      // try it to get the best so far.
      wA = delta(wA, dA, brackets.ax);
    }

    double alpha;    
    if(foundBracket)
    {
      // Minimize along the line.
      LineMinimizationResults res = lineMinimize(wA, dA, brackets, f);
      oldV = val;
      val = res.value;
      wA = res.pos;

      // Get some info from line minimization results
      alpha = res.alpha;
      gA = res.gradient;

      /+
      version(unittest)
      {
        writeln(res.toString);
      }
      +/
    }
    else // Couldn't bracket, but use the best value we got from trying.
    {
      f.evaluate(wA, true);
      oldV = val;
      val = f.value;
      gA = f.gradient;
      alpha = 0.0;

      /+
      version(unittest)
      {
        write("Failed to bracket - so -");
        writeln(brackets.toString);
      }
      +/
    }

    // Update the error averages to check stopping criteria
    deltaV[idx] = abs(val - oldV);
    idx = idx < XX - 1 ? idx + 1 : 0; // cycle through this array
    avgDeltaV = 0.0;
    foreach(i; 0 .. XX)avgDeltaV += deltaV[i];
    avgDeltaV /= XX;
    
    // Check stopping criteria
    if(iter >= maxIt){
      // throw the best we have so far in there
      startPos = wA;

      // Get out of here and signal failure!
      throw new FailureToConverge(
        format("Failure to converge: achieved %g tol after %d iteratons" ~
               " of %d max, goal was %g.", 
               avgDeltaV, iter, maxIt, minDeltaV),
        iter, avgDeltaV, wA, val);
    }
    
    if(avgDeltaV < minDeltaV) break; // Convergence!!!

    g = CVector(gA);
    
    // Update parameters for calculating new G
    if(alpha == 0.0) {
      G = Matrix.identity(gA.length);
    }
    else{
      Matrix p = d * alpha; // equals w - oldw from equations
      Matrix v = g - oldg;
      const double tmp = ((v.Tv * G * v)[0,0]);
      Matrix u = p / ((p.Tv * v)[0,0]) - G * v / tmp;
      const Matrix term1 = p.Tv % p / ((p.Tv * v)[0,0]);
      const Matrix term2 = ((G * v) % v.Tv) * G / tmp;
      const Matrix term3 = tmp * (u % u.Tv);
      G += term1 - term2 + term3;
    }
  }

  startPos = wA;
}

///
unittest
{
  // bfgsMinimize

  /*
   * Imported function in the unittest version of func.d with an absolute 
   * minimum at (x=1, y=2, z=3).
   */
  AnotherFunction af = new AnotherFunction;

  // Start at point (-10,100,1000), just cause
  double[] startPoint = [-10.0, 100.0, 1000.0];
  bfgsMinimize(af, startPoint, 1000, 1.0e-12);
  assert(approxEqual(startPoint, [1.0, 2.0, 3.0]));

  /*
   * Imported function in the unittest version of func.d with an absolute 
   * minimum at (x=0, y=0, z=0, .....).
   */
  Func sm = new SquareMachine;
  startPoint = [-10.0, 100.0, 100.0, 1.0, 20.0, 567.88888];
  try{
    bfgsMinimize(sm, startPoint, 1000, 1.0e-12);
    assert(approxEqual(startPoint, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]));
  }
  catch(FailureToConverge e){
    writeln(e.msg);
    assert(0);
  }

  /*
   * Imported function in the unittest version of func.d with an absolute 
   * minimum at infinity and no local minima.
   */
  Func nf = new ANegativeFunction;
  startPoint = [-0.01, 0.01, 0.02];
  assertThrown!FailureToConverge(bfgsMinimize(nf, startPoint, 1000, 1.0e-12));
}
