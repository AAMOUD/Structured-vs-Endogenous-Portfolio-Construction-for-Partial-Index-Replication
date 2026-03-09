"""Evaluation metrics utilities."""
import numpy as np

def tracking_error(portfolio_returns, index_returns):

    return np.std(portfolio_returns - index_returns)


def annualized_te(te):

    return te * np.sqrt(252)


def turnover(w_old, w_new):

    return np.sum(np.abs(w_old - w_new))