from sklearn.ensemble import RandomForestClassifier
from .base_selector import FeatureSelectorBase
import pandas as pd

class RFImportanceSelector(FeatureSelectorBase):
    """
    Chọn đặc trưng dựa trên độ quan trọng (feature importance)
    từ mô hình Random Forest.
    """

    def __init__(self, n_estimators=100, k=10):
        super().__init__()
        self.k = k
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        importances = pd.Series(self.model.feature_importances_, index=X.columns)
        self.selected_features = importances.sort_values(ascending=False).head(self.k).index

    def transform(self, X):
        return X[self.selected_features]