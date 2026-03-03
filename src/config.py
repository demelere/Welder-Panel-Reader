import yaml
from pydantic import BaseModel
from typing import Tuple, Union

class ROIConfig(BaseModel):
    x: float
    y: float
    w: float
    h: float

class AppConfig(BaseModel):
    camera_device: Union[int, str]
    width: int
    height: int
    roi_amps: ROIConfig
    roi_volts: ROIConfig
    inference_hz: float
    confidence_threshold: float
    amps_range: Tuple[float, float]
    volts_range: Tuple[float, float]

def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
