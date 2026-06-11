#!/bin/bash
# Build and run the omni-verify executable with a working Metal toolchain.
#
#   ./Scripts/run-verify.sh <args passed to omni-verify...>
#   ./Scripts/run-verify.sh chatverify "$CHAT_DIR" Fixtures/chat_fixtures.json
#   ./Scripts/run-verify.sh "$MODEL_DIR" Fixtures/text_fixtures.json
#
# Why this exists instead of `swift run omni-verify`:
#   mlx-swift's Metal kernels (default.metallib) are produced by Xcode's PrepareMetalShaders build
#   step, which SwiftPM's CLI (`swift build`/`swift run`) does not run - so a plain `swift run`
#   crashes with "Failed to load the default metallib". Building via xcodebuild (as the app and the
#   test bundle already do) generates the metallib next to the binary. This mirrors run-tests.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

# xcodebuild needs a full Xcode (the metallib step does not run under CommandLineTools). If the
# active developer dir is CLT, fall back to an installed Xcode(-beta) via DEVELOPER_DIR.
if ! xcode-select -p 2>/dev/null | grep -q "Xcode"; then
  for x in /Applications/Xcode.app /Applications/Xcode-beta.app; do
    if [ -d "$x/Contents/Developer" ]; then export DEVELOPER_DIR="$x/Contents/Developer"; break; fi
  done
fi

DD=".build/xcode-rel"   # reuse the app/test derived data so the Metal toolchain / packages are warm
ART="$PWD/$DD/SourcePackages/artifacts/swift-tokenizers/TokenizersRust/TokenizersRust.artifactbundle"

# The generated Omni.xcodeproj shadows the SwiftPM package for xcodebuild; move it aside if present.
moved=0
if [ -d Omni.xcodeproj ]; then mv Omni.xcodeproj "/tmp/Omni.xcodeproj.bak.$$"; moved=1; fi
restore() { [ "$moved" = 1 ] && mv "/tmp/Omni.xcodeproj.bak.$$" Omni.xcodeproj || true; }
trap restore EXIT

xcodebuild -resolvePackageDependencies -scheme Omni-Package -derivedDataPath "$DD" >/dev/null

# MTL_FAST_MATH=NO builds mlx-swift's Metal kernels with precise floating point, matching the
# precise-math kernels in pip's mlx (which generated the reference fixtures). With fast math the
# forward pass still produces the same argmax tokens, but the full-logit cosine drifts ~0.2% - so we
# verify against precise kernels. (The shipped app may keep the default fast math for speed; this
# flag only affects the numeric-parity build.)
xcodebuild build -scheme Omni-Package -destination 'platform=macOS' -derivedDataPath "$DD" \
  -configuration Release MTL_FAST_MATH=NO \
  OTHER_SWIFT_FLAGS="\$(inherited) -Xcc -fmodule-map-file=$ART/include/module.modulemap -Xcc -I$ART/include" \
  OTHER_LDFLAGS="\$(inherited) $ART/apple-macos/libtokenizers_rust.a" >/dev/null

PROD="$DD/Build/Products/Release"
BIN="$PROD/omni-verify"
[ -x "$BIN" ] || { echo "omni-verify not found at $BIN"; exit 1; }

DYLD_FRAMEWORK_PATH="$PROD" DYLD_LIBRARY_PATH="$PROD" "$BIN" "$@"
