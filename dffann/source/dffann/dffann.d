module dffann.dffann;

import std.compiler;
import std.conv;
import std.range;

pragma(msg, "Compiler is " ~ name);
pragma(msg, "Compiler id: " ~to!string(vendor));
pragma(msg, "Compiler version: " ~ to!string(version_major)~
  "."~to!string(version_minor));

static if( vendor == Vendor.llvm &&  version_minor <= 68)
{
  pragma(msg, "    Need to add evenChunks.");

  struct EvenChunks(Source)
      if (isForwardRange!Source && hasLength!Source)
  {
      /// Standard constructor
      this(Source source, size_t chunkCount)
      {
          assert(chunkCount != 0 || source.empty, "Cannot create EvenChunks with a zero chunkCount");
          _source = source;
          _chunkCount = chunkCount;
      }

      /// Forward range primitives. Always present.
      @property auto front()
      {
          assert(!empty);
          return _source.save.take(_chunkPos(1));
      }

      /// Ditto
      void popFront()
      {
          assert(!empty);
          _source.popFrontN(_chunkPos(1));
          _chunkCount--;
      }

      /// Ditto
      @property bool empty()
      {
          return _source.empty;
      }

      /// Ditto
      @property typeof(this) save()
      {
          return typeof(this)(_source.save, _chunkCount);
      }

      /// Length
      @property size_t length()
      {
          return _chunkCount;
      }
      //Note: No point in defining opDollar here without slicing.
      //opDollar is defined below in the hasSlicing!Source section

      static if (hasSlicing!Source)
      {
          /**
          Indexing, slicing and bidirectional operations and range primitives.
          Provided only if $(D hasSlicing!Source) is $(D true).
           */
          auto opIndex(size_t index)
          {
              assert(index < _chunkCount, "evenChunks index out of bounds");
              return _source[_chunkPos(index) .. _chunkPos(index+1)];
          }

          /// Ditto
          typeof(this) opSlice(size_t lower, size_t upper)
          {
              assert(lower <= upper && upper <= length, "evenChunks slicing index out of bounds");
              return evenChunks(_source[_chunkPos(lower) .. _chunkPos(upper)], upper - lower);
          }

          /// Ditto
          @property auto back()
          {
              assert(!empty, "back called on empty evenChunks");
              return _source[_chunkPos(_chunkCount - 1) .. $];
          }

          /// Ditto
          void popBack()
          {
              assert(!empty, "popBack() called on empty evenChunks");
              _source = _source[0 .. _chunkPos(_chunkCount - 1)];
              _chunkCount--;
          }
      }

  private:
      Source _source;
      size_t _chunkCount;

      size_t _chunkPos(size_t i)
      {
          /*
              _chunkCount = 5, _source.length = 13:

                 chunk0
                   |   chunk3
                   |     |
                   v     v
                  +-+-+-+-+-+   ^
                  |0|3|.| | |   |
                  +-+-+-+-+-+   | div
                  |1|4|.| | |   |
                  +-+-+-+-+-+   v
                  |2|5|.|
                  +-+-+-+

                  <----->
                    mod

                  <--------->
                  _chunkCount

              One column is one chunk.
              popFront and popBack pop the left-most
              and right-most column, respectively.
          */

          auto div = _source.length / _chunkCount;
          auto mod = _source.length % _chunkCount;
          auto pos = i <= mod
              ? i   * (div+1)
              : mod * (div+1) + (i-mod) * div
          ;
          //auto len = i < mod
          //    ? div+1
          //    : div
          //;
          return pos;
      }
  }

  /// Ditto
  EvenChunks!Source evenChunks(Source)(Source source, size_t chunkCount)
  if (isForwardRange!Source && hasLength!Source)
  {
      return typeof(return)(source, chunkCount);
  }

}
