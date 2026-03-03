import argparse
import sys
import cv2
from src.vlm_moondream import MoondreamWrapper
from src.config import load_config
from src.rois import crop_roi

def main():
    parser = argparse.ArgumentParser(description="Test Moondream inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", required=True, help="Path to config YAML (for ROIs)")
    args = parser.parse_args()

    config = load_config(args.config)
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Failed to load image: {args.image}", file=sys.stderr)
        sys.exit(1)

    vlm = MoondreamWrapper()

    img_amps = crop_roi(image, config.roi_amps)
    img_volts = crop_roi(image, config.roi_volts)
    
    print("Inferring Amps...")
    res_amps = vlm.infer(img_amps)
    print("Amps Result:", res_amps.model_dump_json(indent=2) if res_amps else "null")

    print("\nInferring Volts...")
    res_volts = vlm.infer(img_volts)
    print("Volts Result:", res_volts.model_dump_json(indent=2) if res_volts else "null")

if __name__ == "__main__":
    main()

