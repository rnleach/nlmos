# nlmos

A collection of modules with the ultimate purpose of making Model Output Statistics (MOS) programs with non-linear techniques. Current emphasis is on feed forward neural networks.

## Data sources
Some module will show up in this repository related to parsing [Bufkit sounding files](www.wdtb.noaa.gov/tools/BUFKIT/). For now, this is the primary source of model data that will be used for generating data. I'll also be using various sources of observational data from any source I can find archived weather observations at locations with soundings.

## Data Management

Currently I have a [MySQL](http://dev.mysql.com/downloads/) database set up to store my weather data, so some code in the repository may reflect that and not really be portable to other use cases. I'll try to keep the modules related to data representation and data management seperate.

## Purpose

This project and code is done as a hobby for my own learning about programming, statistics, neural networks, and whatever else I feel like learning about. It is not intended for any kind of production code. 

## License

I haven't really licensed any of the code. I don't care if you use it, but since this is a hobby I haven't given much attention to others using my code. Since this is a publicly available repository, I should mention that some of the code in the [numeric](numeric/source/) module was ported from [Numerical Recipes, The Art of Scientific Computing](http://numerical.recipes/). I have annotated where in the code I ported and modified the routines from the books (of which I own two different editions). Be sure to read their licensing BEFORE using that code. It is very awkward.
