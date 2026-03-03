# Panel Reader

Reads welding amperage and voltage from a fixed instrument panel display using Continuity Camera and Moondream on macOS.

## Notes

- Engineering decisions and troubleshooting notes: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Setup

Requires Python 3.10+ and `uv`.

```bash
# Install dependencies using uv
uv sync
```

### Running the Moondream Server

Because we are doing entirely local inference, you must run the `moondream-station` server in the background (or in a separate terminal) before running this app:

```bash
# In a separate terminal session, start the station:
uv run moondream-station
```
Ensure you have a model downloaded and active inside the station (e.g. `moondream-2`). The station will launch a local server on port `2020` which this application connects to via the `moondream` Python client.

## macOS Continuity Camera Selection

To use your iPhone as a webcam on macOS:
1. Ensure your iPhone and Mac are on the same Wi-Fi network and signed into the same Apple ID.
2. List AVFoundation devices with ffmpeg:

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

3. Set `camera_device` in your config to either:
- the numeric index (for example `3`), or
- the exact device name (for example `Stephen's iPhone Camera`).

Some OpenCV builds on macOS cannot open higher device indices directly. This app now falls back to ffmpeg capture for those cases.

## Usage

```bash
# Run with ROI drawing to help you align the camera
python -m src.app --config configs/dynasty350.yaml --show-rois

# Log output to a file
python -m src.app --config configs/dynasty350.yaml --save-log output.jsonl
```

## Adjusting ROIs

The `configs/dynasty350.yaml` file defines the Region of Interest (ROI) for the amps and volts displays. The values `x`, `y`, `w`, `h` are normalized coordinates (0.0 to 1.0) relative to the image width and height.

1. Run the app with `--show-rois`.
2. Physically align the camera as best as possible.
3. Tweak the `x`, `y`, `w`, `h` values in the config file until the boxes perfectly enclose the 7-segment displays.
4. Restart the app to see the updated boxes.

## Future Optimization (Jetson Orin Nano)

The current architecture runs inference asynchronously to prevent blocking the camera feed. To deploy this to a Jetson Orin Nano:
1. Ensure `moondream-station` can utilize Jetson's GPU (e.g., via TensorRT or ONNX Runtime with CUDA execution provider).
2. The current 8 Hz inference rate should be achievable on an Orin Nano, but if performance is lacking, consider swapping Moondream for a deterministic 7-segment decoder or lightweight OCR (like Tesseract or a small custom CNN).

### Replacing Moondream

The application is modular. To replace Moondream with a faster, deterministic approach:
1. Create a new class (e.g., `DeterministicDecoder`) with an `infer(self, image: np.ndarray) -> Optional[InferenceResult]` method.
2. Update `src/app.py` to instantiate and use your new decoder instead of `MoondreamWrapper`.

## Testing inference on a single image

If you have a saved frame, you can run a single inference pass without using the camera:

```bash
python -m src.test_inference --image test_frame.jpg --config configs/dynasty350.yaml
```
