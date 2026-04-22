from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from .base_selector import FeatureSelectorBase


class RFESelector(FeatureSelectorBase):
    """
    Recursive Feature Elimination:
    Lặp lại việc loại bỏ feature yếu nhất dựa trên model.
    """

    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.model = LogisticRegression(max_iter=1000)
        self.selector = RFE(estimator=self.model, n_features_to_select=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.support_]

    def transform(self, X):
        return X[self.selected_features]
