import cv2
import platform
import re
import shutil
import subprocess
from typing import Union
import numpy as np

class Camera:
    def __init__(self, device_id: Union[int, str], width: int, height: int):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
        self.ffmpeg_proc = None
        parsed_device = self._parse_device_id(device_id)

        # On macOS, prefer AVFoundation so Continuity Camera indices resolve.
        candidates = []
        if platform.system() == "Darwin":
            candidates.append((parsed_device, cv2.CAP_AVFOUNDATION))
        candidates.append((parsed_device, cv2.CAP_ANY))

        for source, backend in candidates:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()

        if self.cap is None and platform.system() == "Darwin":
            # Some OpenCV builds on macOS cannot open AVFoundation devices > 1.
            self.ffmpeg_proc = self._open_ffmpeg_avfoundation(parsed_device, width, height)

        if self.cap is None:
            if self.ffmpeg_proc is None:
                raise RuntimeError(
                    f"Could not open camera device {device_id}. "
                    "On macOS, try a Continuity Camera index from "
                    "`ffmpeg -f avfoundation -list_devices true -i \"\"` "
                    "(e.g. 3 for iPhone Camera), or set camera_device to the exact device name."
                )
            
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read a frame to ensure it works
        frame = self.read_frame()
        if frame is None:
            self.release()
            raise RuntimeError(
                f"Could not read from camera device {device_id}. "
                "If using Continuity Camera on macOS, try camera_device as the exact ffmpeg device name."
            )

    @staticmethod
    def _parse_device_id(device_id: Union[int, str]) -> Union[int, str]:
        if isinstance(device_id, int):
            return device_id
        try:
            return int(device_id)
        except (TypeError, ValueError):
            return device_id

    def _open_ffmpeg_avfoundation(self, device: Union[int, str], width: int, height: int):
        if shutil.which("ffmpeg") is None:
            return None

        video_device = self._resolve_avfoundation_device_name(device)
        if video_device is None:
            return None

        input_spec = f"{video_device}:none"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-i",
            input_spec,
            "-vf",
            f"scale={width}:{height}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8,
            )
        except Exception:
            return None
        return proc

    def _resolve_avfoundation_device_name(self, device: Union[int, str]) -> Union[str, None]:
        if isinstance(device, str):
            return device

        cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception:
            return None

        # ffmpeg prints AVFoundation device list to stderr.
        text = res.stderr or ""
        pattern = re.compile(r"\[AVFoundation indev @ [^\]]+\] \[(\d+)\] (.+)")
        device_map = {}
        in_video_section = False
        for line in text.splitlines():
            if "AVFoundation video devices:" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices:" in line:
                in_video_section = False
                continue
            if not in_video_section:
                continue
            match = pattern.search(line)
            if match:
                idx = int(match.group(1))
                name = match.group(2).strip()
                device_map[idx] = name

        # Prefer main iPhone Continuity Camera and avoid Desk View.
        for name in device_map.values():
            lowered = name.lower()
            if "iphone camera" in lowered and "desk view" not in lowered:
                return name

        # Fallback to explicit index if no preferred iPhone camera was found.
        return device_map.get(device)

    def read_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            return frame if ret else None

        if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
            return None

        frame_size = self.width * self.height * 3
        raw = self._read_exact(frame_size)
        if raw is None:
            return None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def _read_exact(self, size: int) -> Union[bytes, None]:
        if self.ffmpeg_proc is None or self.ffmpeg_proc.stdout is None:
            return None
        chunks = []
        bytes_read = 0
        while bytes_read < size:
            chunk = self.ffmpeg_proc.stdout.read(size - bytes_read)
            if not chunk:
                return None
            chunks.append(chunk)
            bytes_read += len(chunk)
        return b"".join(chunks)
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.ffmpeg_proc is not None:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=1.0)
            except Exception:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None
