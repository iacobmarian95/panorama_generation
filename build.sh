rm -r debug
mkdir debug
cd debug
cmake -G "MinGW Makefiles"  ..
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
