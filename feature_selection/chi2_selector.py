from sklearn.feature_selection import SelectKBest, chi2
from .base_selector import FeatureSelectorBase

class Chi2Selector(FeatureSelectorBase):
    """
    Chọn K đặc trưng tốt nhất dựa trên kiểm định Chi-Square.
    Phù hợp với dữ liệu không âm (>=0).
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.selector = SelectKBest(score_func=chi2, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()]

    def transform(self, X):
        return X[self.selected_features]