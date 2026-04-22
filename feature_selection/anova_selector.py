from sklearn.feature_selection import SelectKBest, f_classif
from .base_selector import FeatureSelectorBase

class ANOVASelector(FeatureSelectorBase):
    """
    Chọn K đặc trưng dựa trên ANOVA F-test.
    Đánh giá sự khác biệt trung bình giữa các class.
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()]

    def transform(self, X):
        return X[self.selected_features]