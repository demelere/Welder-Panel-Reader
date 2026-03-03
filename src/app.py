import argparse
import time
import json
import cv2
import sys
import threading

from src.config import load_config
from src.camera import Camera
from src.rois import crop_roi, draw_roi
from src.vlm_moondream import MoondreamWrapper
from src.smoothing import Debouncer
from src.overlay import draw_overlay

def main():
    parser = argparse.ArgumentParser(description="Panel Reader")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--show-rois", action="store_true", help="Draw ROI rectangles on the output window")
    parser.add_argument("--no-roi", action="store_true", help="Analyze the full frame instead of cropping ROIs")
    parser.add_argument("--save-log", type=str, help="Save JSON lines output to file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Open log file if specified
    log_file = open(args.save_log, "a") if args.save_log else None

    # Initialize components
    cam = Camera(config.camera_device, config.width, config.height)
    vlm = MoondreamWrapper()
    
    debounce_amps = Debouncer(required_consecutive=2)
    debounce_volts = Debouncer(required_consecutive=2)
    
    inference_interval = 1.0 / config.inference_hz
    last_inference_time = 0
    
    latest_frame = None
    frame_lock = threading.Lock()
    
    # State for overlay
    current_amps = None
    current_volts = None
    
    running = True

    def inference_loop():
        nonlocal current_amps, current_volts, latest_frame, running
        while running:
            start_time = time.time()
            
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
                
            if frame is not None:
                if args.no_roi:
                    # Use full frame for both
                    img_amps = frame
                    img_volts = frame
                else:
                    # Crop ROIs
                    img_amps = crop_roi(frame, config.roi_amps)
                    img_volts = crop_roi(frame, config.roi_volts)
                
                # Infer Amps
                res_amps = vlm.infer(img_amps)
                # Infer Volts
                res_volts = vlm.infer(img_volts)
                
                # Validate and apply constraints for Amps
                raw_amps = None
                amps_conf = 0.0
                if res_amps and res_amps.unit == "A" and res_amps.confidence >= config.confidence_threshold:
                    if res_amps.value is not None and config.amps_range[0] <= res_amps.value <= config.amps_range[1]:
                        raw_amps = res_amps.value
                        amps_conf = res_amps.confidence
                        
                # Validate and apply constraints for Volts
                raw_volts = None
                volts_conf = 0.0
                if res_volts and res_volts.unit == "V" and res_volts.confidence >= config.confidence_threshold:
                    if res_volts.value is not None and config.volts_range[0] <= res_volts.value <= config.volts_range[1]:
                        raw_volts = res_volts.value
                        volts_conf = res_volts.confidence
                
                # Temporal smoothing
                smoothed_amps = debounce_amps.update(raw_amps)
                smoothed_volts = debounce_volts.update(raw_volts)
                
                current_amps = smoothed_amps
                current_volts = smoothed_volts
                
                # Output JSON
                out_data = {
                    "timestamp": time.time(),
                    "amps": smoothed_amps,
                    "amps_conf": float(amps_conf),
                    "volts": smoothed_volts,
                    "volts_conf": float(volts_conf)
                }
                out_json = json.dumps(out_data)
                print(out_json, flush=True)
                if log_file:
                    log_file.write(out_json + "\n")
                    log_file.flush()
            
            elapsed = time.time() - start_time
            sleep_time = max(0, inference_interval - elapsed)
            time.sleep(sleep_time)

    # Start inference thread
    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print(f"Starting camera feed. Press 'q' to quit.", file=sys.stderr)

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                print("Failed to read frame from camera.", file=sys.stderr)
                time.sleep(0.1)
                continue
                
            with frame_lock:
                latest_frame = frame
                
            display_frame = frame.copy()
            
            if args.show_rois and not args.no_roi:
                draw_roi(display_frame, config.roi_amps, color=(0, 255, 0), label="Amps ROI")
                draw_roi(display_frame, config.roi_volts, color=(255, 0, 0), label="Volts ROI")
                
            draw_overlay(display_frame, current_amps, current_volts)
            
            cv2.imshow("Panel Reader Preview", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        cam.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()
