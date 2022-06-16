class BaseAlgorithm:
    def fit(self, df):
        pass

    def predict(self, current_data=None, n=1):
        pass

    def predict_step(self, step=1, context=None):
        pass
