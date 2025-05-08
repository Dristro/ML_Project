import pickle
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin

class Phase3Pipeline(BaseEstimator, RegressorMixin):

    def __init__(self, model=None, scaler=None):
 
        self.model = model
        self.scaler = scaler

    def fit(self, X, y):
        # Fit scaler and model together
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        # Apply scaler then model
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path):
        """
        Save the entire pipeline to disk.
        """
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[INFO] Phase-3 pipeline saved to {path.resolve()}")

    @classmethod
    def load(cls, path):
        """
        Load a saved Phase-3 pipeline.
        """
        path = Path(path)
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"[INFO] Phase-3 pipeline loaded from {path.resolve()}")
        return pipeline

