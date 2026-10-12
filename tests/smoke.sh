#!/usr/bin/env bash
set -euo pipefail
# A small deterministic PCM signal for exercising native encode/decode.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENC="${ENC:-$ROOT/build/lxvoxel-enc}"
DEC="${DEC:-$ROOT/build/lxvoxel-dec}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT
python3 - "$DIR/source.f32" <<'PYPCM'
import math, struct, sys
with open(sys.argv[1], 'wb') as out:
    for n in range(48000):
        l = 0.20 * math.sin(2 * math.pi * 440 * n / 48000)
        r = 0.15 * math.sin(2 * math.pi * 880 * n / 48000)
        out.write(struct.pack('ff', l, r))
PYPCM
"$ENC" "$DIR/source.f32" --kbps 160 "$DIR/reconstructed.f32" "$DIR/test.vox"
"$DEC" "$DIR/test.vox" "$DIR/decoded.f32"
test -s "$DIR/test.vox"
test -s "$DIR/decoded.f32"
python3 - "$DIR/decoded.f32" <<'PYCHK'
import math, struct, sys
raw = open(sys.argv[1], 'rb').read()
assert len(raw) % 4 == 0 and len(raw) > 0
assert all(math.isfinite(x[0]) for x in struct.iter_unpack('f', raw)), 'nonfinite decoded sample'
print('Native encode/decode smoke test passed; decoded float samples:', len(raw)//4)
PYCHK
