import cv2
import numpy as np
from src.config import ROIConfig

def get_roi_rect(image_shape, roi: ROIConfig):
    """Convert normalized ROI coordinates to absolute pixel coordinates."""
    h_img, w_img = image_shape[:2]
    x1 = int(roi.x * w_img)
    y1 = int(roi.y * h_img)
    w = int(roi.w * w_img)
    h = int(roi.h * h_img)
    x2 = min(x1 + w, w_img)
    y2 = min(y1 + h, h_img)
    return x1, y1, x2, y2

def crop_roi(image: np.ndarray, roi: ROIConfig) -> np.ndarray:
    """Crop the given ROI from the image."""
    x1, y1, x2, y2 = get_roi_rect(image.shape, roi)
    return image[y1:y2, x1:x2]

def draw_roi(image: np.ndarray, roi: ROIConfig, color=(0, 255, 0), label=""):
    """Draw an ROI rectangle and label on the image."""
    x1, y1, x2, y2 = get_roi_rect(image.shape, roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
