"""Correlation-based model implementation."""
from .base_model import BaseModel

class CorrelationModel(BaseModel):

    def fit(self, R, index_returns):

        corr = R.corrwith(index_returns)

        selected = corr.sort_values(ascending=False).head(self.K).index

        selected = list(selected)
        self.selected_assets = selected
        self.weights = self.refit_long_only_weights(R, index_returns, selected)