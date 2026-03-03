from typing import Optional

class Debouncer:
    def __init__(self, required_consecutive: int = 2):
        self.required_consecutive = required_consecutive
        self.history = []
        self.current_stable_value = None

    def update(self, new_value: Optional[float]) -> Optional[float]:
        self.history.append(new_value)
        if len(self.history) > self.required_consecutive:
            self.history.pop(0)

        # Check if all values in history are the same and we have enough history
        if len(self.history) == self.required_consecutive and all(v == self.history[0] for v in self.history):
            self.current_stable_value = self.history[0]
            
        return self.current_stable_value
