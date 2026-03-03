# Decisions Log

## 2026-03-03: Continuity Camera capture strategy on macOS

### Context
- `ffmpeg -f avfoundation -list_devices true -i ""` could see iPhone Continuity Camera devices.
- OpenCV failed to open those same devices by index (`out device of bound (0-1)` / `(0-0)` for indices like `3`).
- Result in app: `RuntimeError: Could not open camera device ...`.

### Decision
- Keep OpenCV as first attempt for normal webcams.
- Add macOS fallback to ffmpeg AVFoundation capture when OpenCV cannot open the selected camera.
- Support `camera_device` as either:
  - numeric index (example: `3`), or
  - exact ffmpeg device name (example: `Stephen's iPhone Camera`).

### Why
- This bypasses OpenCV backend limitations for Continuity Camera on some macOS/OpenCV builds.
- It keeps existing behavior for users where OpenCV camera capture works.

### Implementation Notes
- `src/camera.py`
  - Added backend selection and ffmpeg fallback process.
  - Added AVFoundation device-name resolution for numeric IDs.
  - Added robust frame assembly for raw ffmpeg pipe reads (`_read_exact`) to avoid partial-frame corruption/freeze.
  - Routed ffmpeg stderr to `DEVNULL` to avoid pipe backpressure stalls.
  - Improved release handling to terminate/kill ffmpeg cleanly.
- `src/config.py`
  - `camera_device` changed from `int` to `Union[int, str]`.
- `README.md`
  - Updated camera selection instructions and documented fallback behavior.

### Operational Guidance
- Prefer exact camera name in config for stability:
  - `camera_device: "Stephen's iPhone Camera"`
- If preview is unstable, lower resolution temporarily (for example `640x480`) and verify connection.

### Non-blocking warning
- `NSCameraUseContinuityCameraDeviceType` warning from CLI tools is expected in this setup and not the root cause of the failure above.
