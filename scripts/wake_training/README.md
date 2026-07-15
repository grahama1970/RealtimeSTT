# Hey Embry OpenWakeWord model gate

This patch creates a real, unqualified `hey_embry_v1.onnx` through the pinned official OpenWakeWord trainer. It never writes a placeholder classifier and never treats transcript substring matching as wake authority.

Pinned upstream sources:

- OpenWakeWord commit `368c03716d1e92591906a84949bc477f3a834455`
- David Scripka's compatible Piper sample-generator fork commit `f1988a4d54eddb23d99e86f0adfef6226a85acc7`
- Python `3.12.3`, SciPy `1.14.1`, PyTorch `2.10.0+cu128`

The first synthetic build proves only real Horus WAV generation, official OpenWakeWord augmentation/training, ONNX Runtime loading, and a measured synthetic held-out operating point. It does **not** prove a physical human or Jabra wake callback.

## 1. Start or reuse Horus synthesis

The existing local service is authoritative. Reuse it when healthy:

```bash
curl -fsS http://127.0.0.1:8767/health | jq -e '.status == "ok"'
```

When it is not running, start it from the existing agent-skills service definition:

```bash
cd /home/graham/workspace/experiments/agent-skills

docker compose \
  -f skills/voice-segment-selector/docker/docker-compose.orpheus-infer.yml \
  up -d --build orpheus-infer

curl --retry 60 --retry-delay 2 --retry-connrefused -fsS \
  http://127.0.0.1:8767/health | jq .
```

## 2. Generate a bounded real Horus dataset

Build the trainer image once:

```bash
cd /home/graham/workspace/experiments/RealtimeSTT

docker build \
  -f docker/Dockerfile.wake-trainer \
  -t embry-openwakeword-trainer:hey-embry-v1 .
```

Run the serial generator against the host service. The supplied bounded plan contains 705 serial Horus requests: 200/75/40 positive train/validation/calibration clips and 260/90/40 hard-negative clips. `--network host` is necessary only for the localhost Horus endpoint. The command is resumable and all successful and failed live attempts are written to JSONL.

```bash
ROOT=/mnt/storage12tb/models/embry-openwakeword/hey-embry-v1
mkdir -p "$ROOT"

docker run --rm --gpus device=0 --network host \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/repo:ro" \
  -v "$ROOT:/work" \
  embry-openwakeword-trainer:hey-embry-v1 \
  python3 /workspace/repo/scripts/wake_training/generate_horus_wake_dataset.py \
    --service-url http://127.0.0.1:8767 \
    --plan /workspace/repo/scripts/wake_training/dataset-plan.example.json \
    --output-dir /work/dataset \
    --resume
```

For a quick wiring pilot, append `--max-records 12`. A partial pilot is not trainable. Re-run without the bound to complete the immutable plan.

Additional human or local WAVs can be imported later with `--import-manifest`. Each JSONL row must provide `record_id`, `path`, `split`, `label`, and `source_class`; imported audio is normalized and receipted but remains explicitly non-synthetic.

## 3. Supply official negative/background assets and train

The official training flow requires:

- mono 16 kHz RIR WAVs;
- background WAVs;
- a generic negative feature array such as the official ACAV100M OpenWakeWord features;
- the official false-positive validation feature array.

Mount existing local copies. Missing files fail before training. No dataset download is silently substituted.

```bash
ASSETS=/mnt/storage12tb/datasets/openwakeword
ROOT=/mnt/storage12tb/models/embry-openwakeword/hey-embry-v1

docker run --rm --gpus device=0 \
  -u "$(id -u):$(id -g)" \
  --shm-size 16g \
  -v "$PWD:/workspace/repo:ro" \
  -v "$ROOT:/work" \
  -v "$ASSETS:/assets:ro" \
  embry-openwakeword-trainer:hey-embry-v1 \
  python3 /workspace/repo/scripts/wake_training/train_openwakeword_model.py \
    --dataset-manifest /work/dataset/dataset-manifest.json \
    --output-dir /work/training \
    --rir-dir /assets/rirs/mono-16k \
    --background-dir /assets/background/16k \
    --negative-features /assets/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
    --false-positive-validation-features /assets/validation_set_features.npy \
    --steps 50000 \
    --augmentation-rounds 1 \
    --max-negative-weight 50 \
    --target-fp-per-hour 0.5 \
    --resume
```

The wrapper invokes the pinned official `openwakeword/train.py --augment_clips` and `--train_model`. It consolidates any ONNX external data into one standalone ONNX and fails unless ONNX Runtime can load and execute it.

## 4. Calibrate and package

Calibration uses the `positive_calibration` and `negative_calibration` splits, which are never copied into the training directories. Scores are the maximum native streaming frame score from `Model.predict_clip(..., chunk_size=512)`.

```bash
ROOT=/mnt/storage12tb/models/embry-openwakeword/hey-embry-v1

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/repo" \
  -v "$ROOT:/work" \
  embry-openwakeword-trainer:hey-embry-v1 \
  python3 /workspace/repo/scripts/wake_training/calibrate_openwakeword_model.py \
    --model /work/training/standalone/hey_embry_v1.onnx \
    --training-manifest /work/training/training-manifest.json \
    --dataset-manifest /work/dataset/dataset-manifest.json \
    --min-recall 0.80 \
    --max-false-accept-rate 0.02 \
    --wake-word-buffer-duration 0.1 \
    --output /work/calibration-receipt.json
```

If no measured threshold satisfies both requirements, calibration exits non-zero and packaging is forbidden.

Package the exact five requested files into the repository checkout:

```bash
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace/repo" \
  -v "$ROOT:/work:ro" \
  embry-openwakeword-trainer:hey-embry-v1 \
  python3 /workspace/repo/scripts/wake_training/package_hey_embry_model.py \
    --model /work/training/standalone/hey_embry_v1.onnx \
    --training-manifest /work/training/training-manifest.json \
    --calibration-receipt /work/calibration-receipt.json \
    --output-dir /workspace/repo/models/wake \
    --receipt /workspace/repo/models/hey-embry-package-receipt.json
```

The package directory contains exactly:

```text
models/wake/hey_embry_v1.onnx
models/wake/hey_embry_v1.onnx.sha256
models/wake/hey_embry_v1.model.json
models/wake/hey_embry_v1.training-manifest.json
models/wake/hey_embry_v1.calibration-receipt.json
```

## 5. Load the exact model in the native RealtimeSTT image

```bash
MODEL="$PWD/models/wake/hey_embry_v1.onnx"
MODEL_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
SENSITIVITY="$(jq -er '.selected_sensitivity' models/wake/hey_embry_v1.calibration-receipt.json)"
RUNTIME=/tmp/embry-native-wake-model
rm -rf "$RUNTIME" && mkdir -p "$RUNTIME"
cp -a models/wake/. "$RUNTIME/"

docker run --rm --gpus device=0 \
  -p 127.0.0.1:8020:8020 \
  -e EMBRY_OPENWAKEWORD_MODEL=/models/wake/hey_embry_v1.onnx \
  -e EMBRY_OPENWAKEWORD_MODEL_SHA256="$MODEL_SHA" \
  -e EMBRY_OPENWAKEWORD_MODEL_ID=hey_embry_v1 \
  -e EMBRY_WAKE_SENSITIVITY="$SENSITIVITY" \
  -e EMBRY_PCM_SOCKET=/run/embry/audio/realtimestt-pcm.sock \
  -v "$RUNTIME:/models/wake:ro" \
  -v /run/user/1000/embry-voice:/run/embry/audio \
  embry-realtimestt:native-wake
```

Then inspect:

```bash
curl -i http://127.0.0.1:8020/readiness
```

A readiness PASS proves that the calibrated model hash loaded before recorder construction. It still does not prove that a physical person speaking through the Jabra triggers the native callback. That next gate requires a fresh physical PCM session and a journaled `listener.wake_detected` event whose authority is `openwakeword_native_callback`.
