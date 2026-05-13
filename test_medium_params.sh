#!/bin/bash
#
# Quick test script to run rotation example with medium() parameters
# Usage: ./test_medium_params.sh
#

set -e

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Testing CKKS Rotations with medium() Parameters (N=4096)     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Backup original file
EXAMPLE_FILE="examples/rotations.rs"
BACKUP_FILE="examples/rotations.rs.bak"
cp "$EXAMPLE_FILE" "$BACKUP_FILE"

# Restore on exit
cleanup() {
    echo ""
    echo "Restoring original rotations.rs..."
    mv "$BACKUP_FILE" "$EXAMPLE_FILE"
}
trap cleanup EXIT

# Switch to medium() parameters
echo "Switching to medium() parameters (N=4096, 2048 slots)..."
sed -i 's|// let params = CKKSParams::medium();|let params = CKKSParams::medium();|' "$EXAMPLE_FILE"
sed -i 's|let params = CKKSParams::small();|// let params = CKKSParams::small();|' "$EXAMPLE_FILE"

# Build with release optimization
echo "Building with release optimization..."
cargo build --release --example rotations 2>&1 | grep -E "(Compiling|Finished|error)" || true

# Run the example
echo ""
echo "Running rotation tests with medium() parameters..."
echo "⏱ Estimated time: 3-5 minutes (key generation + tests)"
echo ""

./target/release/examples/rotations

echo ""
echo "✓ Test complete! Original rotations.rs restored."
