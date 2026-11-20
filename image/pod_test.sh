#!/bin/bash
# Willy's acceptance test (CDeep3m-anywhere docker/downloadTest.sh + docker/image_test.sh), run directly
# in a pod: there is no Docker here, so the folder that the Docker test mounts at /data is a normal folder.
#
#   pod_test.sh [data folder]      default: /tmp/cdeep3m-test
#   DEVICE=cpu|vulkan              device for the prediction step (default cpu)
#
# Puts into the data folder (about 3 GB; steps already done are skipped on a rerun):
#   models/SBEM_Mitochondria/{1fm,3fm,5fm}   pre-trained model, downloaded from cildata.crbs.ucsd.edu
#   mito_testsample/                         sample images and labels (copied from the image)
#   train_data/                              augmented training data made from the sample
# then runs the five test steps. A pass ends with "== ALL STEPS PASSED".
set -e -o pipefail
D=$(readlink -f "${1:-/tmp/cdeep3m-test}"); mkdir -p "$D"
case "$D" in *"/data"*|*"/sample"*) echo "pick a folder whose path has no /data or /sample in it" >&2; exit 1 ;; esac
MODEL_URL=https://cildata.crbs.ucsd.edu/media/model_display/50681/SBEM_Mitochondria.tar
echo "data folder: $D"

echo "== 1. model"
if [ -f "$D/models/SBEM_Mitochondria/1fm/deploy.prototxt" ]; then echo "already there"; else
  tmp="$D/.model_tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
  curl -fL --retry 3 -o "$tmp/model.tar" "$MODEL_URL"
  tar xf "$tmp/model.tar" -C "$tmp"
  src=$(dirname "$(dirname "$(find "$tmp" -path '*/1fm/deploy.prototxt' | head -n 1)")")
  [ -d "$src/1fm" ] || { echo "no 1fm/deploy.prototxt in the archive" >&2; exit 2; }
  mkdir -p "$D/models"; rm -rf "$D/models/SBEM_Mitochondria"; mv "$src" "$D/models/SBEM_Mitochondria"; rm -rf "$tmp"
fi
ls "$D/models/SBEM_Mitochondria"

echo "== 2. sample images"
if [ -d "$D/mito_testsample/training/images" ]; then echo "already there"; else
  cp -r "$CDEEP3M2/mito_testsample" "$D/mito_testsample"
fi
ls "$D/mito_testsample"

echo "== 3. training data (augmentation of the sample, a few minutes)"
if ls "$D"/train_data/*.h5 >/dev/null 2>&1; then echo "already there"; else
  mkdir -p "$D/train_data"
  (cd "$CDEEP3M2" && python PreprocessTrainingData.py "$D/mito_testsample/training/images" \
      "$D/mito_testsample/training/labels" 0 "$D/train_data") | tail -n 6
fi
du -sh "$D/train_data"

echo "== 4. test"
# image_test.sh unchanged except for its two Docker mount points
sed -e "s|/sample|$D/mito_testsample|g" -e "s|/data|$D|g" /opt/cdeep3m-anywhere/docker/image_test.sh > "$D/image_test.sh"
DEVICE=${DEVICE:-cpu} bash "$D/image_test.sh" image_test
