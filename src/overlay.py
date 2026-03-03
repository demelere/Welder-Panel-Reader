import cv2
import numpy as np

def draw_overlay(image: np.ndarray, amps: float | None, volts: float | None):
    """Draw parsed values on the live preview."""
    h_img, w_img = image.shape[:2]
    
    # Text setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Amps text
    amps_text = f"Amps: {amps}A" if amps is not None else "Amps: --"
    cv2.putText(image, amps_text, (50, 50), font, font_scale, (0, 0, 255), thickness)
    
    # Volts text
    volts_text = f"Volts: {volts}V" if volts is not None else "Volts: --"
    cv2.putText(image, volts_text, (50, 100), font, font_scale, (255, 0, 0), thickness)
