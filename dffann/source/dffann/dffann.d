/**
 * Feed Forward Artificial Networks (in D).
 *
 * This module is the repository for general purpose utilities in this packages.
 * Specifically any Exceptions will be defined here, and some general purpose
 * functions for generating unittest code.
 *
 * This module should be publicly imported to every other module in this 
 * package.
 * 
 * Author: Ryan Leach
 */
 module dffann.dffann;

 version(unittest){

  // Set up some imports and utility functions for all unit tests.
  public import std.stdio;

  // Generate string for mixin that announces this test.
  string announceTest(in string msg){
    return "
    write(format(\"Testing %s - %5d: %s...\",__FILE__,__LINE__,\"" ~ msg ~"\"));
    scope(exit)writeln(\"done.\");";
  }  
}


