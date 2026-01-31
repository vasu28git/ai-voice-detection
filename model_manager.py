import pickle
import os
import numpy as np
from config import MODEL_PATH, SCALER_PATH


class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        try:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load scaler ({e}). Using identity scaling (raw features).")
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            self.scaler = StandardScaler()
            self.scaler.fit(np.zeros((1, 83)))
    
    def predict(self, features: np.ndarray) -> tuple[str, float, str]:
        scaled_features = self.scaler.transform([features])

        probabilities = self.model.predict_proba(scaled_features)[0]
        confidence = float(max(probabilities))

        if probabilities[1] > 0.5:
            classification = "AI_GENERATED"
            explanation = self._get_ai_explanation(probabilities[1])
        else:
            classification = "HUMAN"
            explanation = self._get_human_explanation(probabilities[0])
        
        return classification, confidence, explanation
    
    def _get_ai_explanation(self, probability: float) -> str:
        if probability > 0.9:
            return "High likelihood of AI generation detected. Unnatural pitch consistency and robotic speech patterns identified."
        elif probability > 0.75:
            return "Strong indicators of AI synthesis detected. Unusual vocal patterns and mechanical rhythm observed."
        else:
            return "Potential AI generation detected. Some synthetic speech characteristics present."
    
    def _get_human_explanation(self, probability: float) -> str:
        """Generate explanation for Human classification."""
        if probability > 0.9:
            return "Confirmed human voice detected. Natural speech patterns, varied pitch, and organic vocal characteristics present."
        elif probability > 0.75:
            return "Likely human voice. Typical human speech characteristics detected with natural variations."
        else:
            return "Probable human voice. Some natural vocal patterns detected."


_model_manager = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
