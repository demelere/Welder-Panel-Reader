import json
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal
import moondream as md

class InferenceResult(BaseModel):
    value: Optional[float]
    unit: Optional[Literal["A", "V"]]
    confidence: float
    raw_text: Optional[str]

class MoondreamWrapper:
    def __init__(self):
        # Initialize moondream client to point to local moondream-station
        # Requires `moondream-station` to be running in another terminal
        self.model = md.vl(endpoint='http://127.0.0.1:2020/v1')
        
        self.prompt = (
            "You are looking at a digital welding machine display. "
            "Identify the amperage (A) and voltage (V) values. "
            "Return ONLY valid JSON.\n"
            "If unreadable or uncertain, set value and unit to null.\n\n"
            "Schema:\n"
            "{\n"
            '  "value": number|null,\n'
            '  "unit": "A"|"V"|null,\n'
            '  "confidence": number,\n'
            '  "raw_text": string|null\n'
            "}"
        )

    def infer(self, image: np.ndarray) -> Optional[InferenceResult]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        try:
            # Query the model using moondream python client
            # The result is a dictionary containing an 'answer' key.
            response = self.model.query(pil_image, self.prompt)
            
            if not response or "answer" not in response:
                return None
                
            response_text = response["answer"]
            
            # Clean up response to ensure it's just JSON
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)
            
            # Validate
            result = InferenceResult(**data)
            return result
            
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            # Silently catch exceptions (e.g., connection refused if station is not running)
            # and return None so the loop can continue.
            return None