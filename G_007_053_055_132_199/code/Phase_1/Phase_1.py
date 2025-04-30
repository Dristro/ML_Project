import pickle
import numpy as np
from sklearn.base import BaseEstimator

class Phase1Pipeline(BaseEstimator):
    def __init__(self, scaler, pca, kmeans_model, cluster_to_stage_map):
        self.scaler = scaler
        self.pca = pca
        self.kmeans_model = kmeans_model
        self.cluster_to_stage_map = cluster_to_stage_map
    
    def transform(self, X, perform_std: bool = False):
        X = self.scaler.transform(X) if perform_std else X
        pc1 = self.pca.transform(X)[:, 0].reshape(-1, 1)
        preds = self.kmeans_model.predict(pc1)
        stages = np.array([self.cluster_to_stage_map[pred] for pred in preds])
        return stages, pc1

    def predict(self, X, return_pc1: bool = False, perform_std: bool = False):
        """
        Args:
            X (np.ndarray): engine sensor readings from the selected sensors
            return_pc1 (bool = False): returns the pc1 data if True
            perform_stf (bool = False): standardize X before clustering
        
        Returns:
            tuple(stages, pc1) or stages
            stages: model predictions (mapping is applied)
            pc1: principle component (we use this for plotting)
        """
        stages, pc1 = self.transform(X)
        if return_pc1:
            return stages, pc1
        return stages


# Create a simple load function (to load model and stuff from saved path)
def load_model_from_path(save_path):
    with open(save_path, "rb") as f:
        model_bundle = pickle.load(f)
        print(f"[INFO] Loaded model-bundle from: {save_path}")
    scaler = model_bundle["scaler"]
    pca = model_bundle["pca"]
    kmeans_model = model_bundle["kmeans_model"]
    cluster_to_stage_map = model_bundle["cluster_to_stage_map"]
    print(f"[INFO] Extracted model components")
    
    # Create model
    model = Phase1Pipeline(
        scaler=scaler,
        pca=pca,
        kmeans_model=kmeans_model,
        cluster_to_stage_map=cluster_to_stage_map,
    )
    print(f"[INFO] Created model instance")
    return model

if __name__ == "__main__":
    model_path = "./stage_prediction_pipeline.pkl"
    loaded_model = load_model_from_path(model_path)
    print(loaded_model)