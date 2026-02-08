#!/bin/bash
set -e

echo "=== Custom macOS Intel Build Script ==="
echo "Setting ORT environment variables..."

# Export ONNX Runtime compilation variables
export ORT_STRATEGY=compile
export ORT_COMPILE_TARGET=x86_64-apple-darwin
export ORT_COMPILE_VERSION=1.19.2

echo "ORT_STRATEGY=$ORT_STRATEGY"
echo "ORT_COMPILE_TARGET=$ORT_COMPILE_TARGET"
echo "ORT_COMPILE_VERSION=$ORT_COMPILE_VERSION"
echo ""
echo "Running Tauri build..."

# Run the tauri build with environment variables explicitly set
npm run tauri build -- --target x86_64-apple-darwin
