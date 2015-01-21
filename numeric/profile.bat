
dub run --build=release --config=mallocFreeProfile --compiler=dmd --arch=x86_64
dub run --build=release --config=mallocFreeParProfile --compiler=dmd --arch=x86_64
dub run --build=release --config=GCProfile --compiler=dmd --arch=x86_64
dub run --build=release --config=GCParProfile --compiler=dmd --arch=x86_64

dub run --build=release --config=mallocFreeProfile --compiler=gdc --arch=x86_64
dub run --build=release --config=mallocFreeParProfile --compiler=gdc --arch=x86_64
dub run --build=release --config=GCProfile --compiler=gdc --arch=x86_64
dub run --build=release --config=GCParProfile --compiler=gdc --arch=x86_64
