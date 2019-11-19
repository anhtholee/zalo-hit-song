"""
============================
Metrics for ZaloAI challenge
============================
Author: Le Anh Tho
"""
import numpy as np
def rmse(targets, predictions):
    """Root mean squared error between the ground truth and the predictions

    Parameters
    ----------
    targets : np.ndarray
        The ground truth
    predictions : np.ndarray
        The predictions

    Returns
    -------
    float
        The output RMSE
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))

