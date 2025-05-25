# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:45:18 2025

@author: weljohn
"""

import pandas as pd
import numpy as np
from iisignature import sig
import iisignature

d = pd.read_csv('c:/temp/standardized_returns.csv',index_col=0)

d.columns

data = d.values


def calc_levy_area(path):
    """
    Calculates Levy area based on the provided path.

    Parameters:
    - path (np.array): Array representing the path.

    Returns:
    - float: Levy area.
    """
    path_sig = sig(path, 2) 
    levy_area = 0.5 * (path_sig[3] - path_sig[4])
    return levy_area
    

def generate_levy_matrix(data):
    """
    Generates Levy lead-lag scoring matrix for the given price panel.

    Returns:
    - pd.DataFrame: Levy lead-lag scoring matrix.
    """
    assets = data.columns
    n = len(assets)
    levy_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            #pair_path = data[[assets[i], assets[j]]].values
            pair_path = data[[assets[i], assets[j]]].cumsum().values
            val = calc_levy_area(pair_path)
            levy_matrix[i, j] = val
            levy_matrix[j, i] = -val
    return(levy_matrix)        



def compute_levy_area_sig(returns1, returns2):

    # Construct 2D path [X_t, Y_t]
    path = np.column_stack([returns1, returns2])
    
    # Compute signature up to level 2
    signature = iisignature.sig(path, 2)
    
    levy_area = 0.5 * (signature[3] - signature[4])
    return levy_area


def compute_levy_area_cum(returns1, returns2):

    # Construct 2D path [X_t, Y_t]
    path = np.column_stack([returns1.cumsum(), returns2.cumsum()])
    
    # Compute signature up to level 2
    signature = iisignature.sig(path, 2)
    
    levy_area = 0.5 * (signature[3] - signature[4])
    return levy_area



def compute_levy_area_pandas_old_wrong(returns1, returns2):
    """
    Matches iisignature's Lévy area calculation exactly for piecewise linear paths.
    """
    path = np.column_stack([returns1, returns2])
    if len(path) < 2:
        return 0.0

    # Get coordinates and increments
    x = path[:, 0]
    y = path[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)

    # Calculate S_12 and S_21 using linear interpolation
    S_12 = np.sum((x[:-1] + 0.5*dx) * dy)  # Trapezoidal rule for ∫X dY
    S_21 = np.sum((y[:-1] + 0.5*dy) * dx)  # Trapezoidal rule for ∫Y dX

    return 0.5 * (S_12 - S_21)


def compute_levy_area_pandas(returns1, returns2):
    path = np.column_stack([returns1, returns2])
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    x_mid = path[:-1, 0] + 0.5*dx
    y_mid = path[:-1, 1] + 0.5*dy
    S_12 = np.sum(x_mid * dy)
    S_21 = np.sum(y_mid * dx)
    return 0.5 * (S_12 - S_21)


def calc_levy_area_deepseek(returns1, returns2):
    path = np.column_stack([returns1, returns2])
    if len(path) < 2:
        return 0.0
    
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    
    x0s = path[:-1, 0]
    y0s = path[:-1, 1]
    
    S12 = np.sum(x0s * dy + 0.5 * dx * dy)
    S21 = np.sum(y0s * dx + 0.5 * dx * dy)
    
    return 0.5 * (S12 - S21)


def compute_levy_area_copilot(returns1, returns2):
    """
    Compute the Lévy area for a 2D path without using iisignature.
    
    This approximation uses the identity:
    
        S_12 = ∫₀ᵀ (X_t - X₀) dY_t     and     S_21 = ∫₀ᵀ (Y_t - Y₀) dX_t,
    
    so that the Lévy area is A = 0.5 * (S_12 - S_21).
    
    Parameters:
      returns1: array-like, sequence for the x-axis.
      returns2: array-like, sequence for the y-axis.
      
    Note: If these are daily increments, consider replacing the following line:
            path = np.column_stack([returns1, returns2])
          with cumulative sums, e.g.,
            path = np.column_stack([np.cumsum(returns1), np.cumsum(returns2)])
          so that each row is a point on the path.
    
    Returns:
      levy_area: the computed Lévy area (float).
    """
    # Construct path: each row a point; adjust here if data are increments.
    path = np.column_stack([returns1, returns2])
    
    # Compute increments between consecutive points
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    
    # Use the left endpoints (all but the last point)
    x_left = path[:-1, 0]
    y_left = path[:-1, 1]
    
    # Compute the approximations for S_12 and S_21:
    # S_12 ~ sum((x_left - x0) * (dy)) and S_21 ~ sum((y_left - y0) * (dx))
    x0 = path[0, 0]
    y0 = path[0, 1]
    S12 = np.sum((x_left - x0) * dy)
    S21 = np.sum((y_left - y0) * dx)
    
    levy_area = 0.5 * (S12 - S21)
    return levy_area

def calc_levy_area_cum_opilot(dx, dy, T=1.0):
    """
    Compute the Lévy area using pandas operations.
    
    Input:
      dx, dy -- daily return sequences (lists or arrays) for two stocks.
      T    -- total time horizon (default 1.0; provided for consistency).
    
    Process:
      1. Convert the returns to pandas Series.
      2. Compute the cumulative sums to form paths: X = cumsum(dx), Y = cumsum(dy).
      3. Approximate the integrals by using the discrete Riemann sum:
         Sum over i = 0 to N-2 of [X[i] * (Y[i+1] - Y[i]) - Y[i] * (X[i+1] - X[i])].
         (Here, X[i] and Y[i] are left endpoints of the increments.)
      4. Divide the sum by 2 to obtain the Lévy area.
    
    Returns:
      The computed Lévy area (float).
    """
    # Convert input lists to pandas Series
    dx = pd.Series(dx)
    dy = pd.Series(dy)
    if len(dx) != len(dy):
        raise ValueError("dx and dy must be the same length.")
    
    # Compute cumulative paths
    X = dx.cumsum()
    Y = dy.cumsum()
    
    # Compute the increments (differences) for the cumulative series.
    # dX[i] = X[i] - X[i-1] for i>=1.
    dX = X.diff().iloc[1:].to_numpy()
    dY = Y.diff().iloc[1:].to_numpy()
    
    # Use the left endpoints for the approximation: index 0 to N-2
    X_left = X.iloc[:-1].to_numpy()
    Y_left = Y.iloc[:-1].to_numpy()
    
    # Compute discrete approximation:
    # Sum_{i=0}^{N-2} [ X[i] * (Y[i+1] - Y[i]) - Y[i] * (X[i+1] - X[i]) ]
    discrete_sum = np.sum(X_left * dY - Y_left * dX)
    return 0.5 * discrete_sum

# Example usage:
if __name__ == "__main__":
    # Suppose we have a simple path (this example assumes the path is cumulative)
    path = np.array([
        [0.0,  0.0],
        [1.0,  1.0],
        [2.0,  3.0]
    ])
    area = calc_levy_area(path)
    print("Computed Lévy area:", area)


r1 = d['BNB']
r2 = d['ADA']

compute_levy_area_sig(r1, r2)
compute_levy_area_pandas(r1, r2)
calc_levy_area_deepseek(r1, r2)
compute_levy_area_copilot(r1, r2)
compute_levy_area_cum(r1, r2)
compute_levy_area_sig(pd.Series(r1).cumsum(), pd.Series(r2).cumsum())
calc_levy_area_cum_opilot(r1, r2)


compute_levy_area_sig([0, 0.5, 0.8, 1.0, -0.12], [0, 0.2, 0.7, 0.5, 0.12])
compute_levy_area_pandas([0, 0.5, 0.8, 1.0, -0.12], [0, 0.2, 0.7, 0.5, 0.012])






levy_matrix_df = pd.DataFrame(levy_matrix, index=assets, columns=assets)
