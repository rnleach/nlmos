/**
 * Some enum types representing different strategies in error function 
 * evaluation and trainer strategies. These are to make template arguments
 * more clear and easily understandable.
 */
module dffann.strategy;

/**
 * Constants to delineate serial or parallelized versions of functions.
 */
enum ParallelStrategy 
{
  serial,     /// Do not parallelize.
  parallel    /// Parallelize the code, usually via a foreach-parallel
}

/**
 * Use mini-batches or evaluate all the data at once.
 */
enum BatchStrategy
{
  batch,      /// Do all of the data at once when training.
  minibatch   /// Use mini-batches and use a subset.
}

/**
 * Randomize the order data points are drawn from a data-set. This is only
 * beneficial when using mini-batches to prevent odd cycles from establishing in
 * the training.
 *
 * This option is usually ignored unless a minibatch strategy is employed.
 */
enum RandomStrategy
{
  random,     /// Randomize the order samples are drawn.
  inOrder     /// Do not randomize the order samples are drawn.
}
