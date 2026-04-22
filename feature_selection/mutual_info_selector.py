from sklearn.feature_selection import SelectKBest, mutual_info_classif
from .base_selector import FeatureSelectorBase

class MutualInfoSelector(FeatureSelectorBase):
    """
    Chọn K đặc trưng dựa trên Mutual Information.
    Đo mức độ phụ thuộc phi tuyến giữa feature và target.
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()]

    def transform(self, X):
        return X[self.selected_features]