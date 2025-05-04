import pickle
from pathlib import Path

class Phase2Pipeline():
    def __init__(self, model):
        """
        Args:
            model: fitted model, XGBoost model trained on SMOTE (enhanced) data.
        """
        self.model = model

    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Model saved to:\n{path.resolve()}")

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path, "rb") as f:
            print(f"[INFO] Loading model from:\n{path.resolve()}")
            return pickle.load(f)

save_path = Path("./stage_classifier_pipeline.pkl")
