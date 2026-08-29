#!/usr/bin/env bash
# Build the golden-test C++ harnesses against the glm5 fork's build tree.
# Usage: tests/golden/build.sh
set -euo pipefail

LLAMACPP=/mnt/ollama/models/llama-cpp-glm5
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
mkdir -p "$BUILD"

CXXFLAGS="-O2 -std=c++17 -I$LLAMACPP/include -I$LLAMACPP/common -I$LLAMACPP/src -I$LLAMACPP/ggml/include"

# collect the static libs
LIBS="$LLAMACPP/build/src/libllama.a $LLAMACPP/build/common/libllama-common.a \
      $LLAMACPP/build/common/libllama-common-base.a \
      $LLAMACPP/build/vendor/cpp-httplib/libcpp-httplib.a"
for lib in "$LLAMACPP"/build/ggml/src/libggml*.a; do
    LIBS="$LIBS $lib"
done

for src in "$HERE"/*.cpp; do
    name="$(basename "$src" .cpp)"
    # skip files that are not harnesses
    if [ "${name#dump_}" = "$name" ] && [ "${name#replay_}" = "$name" ]; then
        continue
    fi
    echo "building $name"
    g++ $CXXFLAGS "$src" -o "$BUILD/$name" $LIBS -lpthread -ldl -lm -lssl -lcrypto -fopenmp
done
echo "done: $BUILD"
