module dffann.dffann;

import std.compiler;
import std.conv;
import std.range;

/// Lowest DMD front end versions supported.
enum lowest_minor_version_dmd_supported = 71;
enum lowest_major_version_dmd_supported = 2;

pragma(msg, "Compiler is " ~ name);
pragma(msg, "Compiler id: " ~to!string(vendor));
pragma(msg, "Compiler version: " ~ to!string(version_major)~
  "."~to!string(version_minor));

enum failureString = "D language versions older than " ~ 
  to!string(lowest_major_version_dmd_supported) ~ "." ~ 
  to!string(lowest_minor_version_dmd_supported) ~ " not supported.";

static assert(version_major >= lowest_major_version_dmd_supported &&
  version_minor >= lowest_minor_version_dmd_supported, failureMsg);

