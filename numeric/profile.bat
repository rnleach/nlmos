
dub run --build=release-nobounds --config=ParProfile --compiler=dmd
dub run --build=release-nobounds --config=SerialProfile --compiler=dmd

dub run --build=release-nobounds --config=ParProfile --compiler=ldc2
dub run --build=release-nobounds --config=SerialProfile --compiler=ldc2
