from abc import ABC, abstractmethod

class FeatureSelectorBase(ABC):
    """
    Base class cho tất cả các phương pháp chọn đặc trưng.
    Định nghĩa interface chung: fit, transform, fit_transform.
    """

    def __init__(self):
        self.selected_features = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        """
        Fit selector trên dữ liệu train và trả về dữ liệu đã được chọn đặc trưng.
        """
        self.fit(X, y)
        return self.transform(X)