import numpy as np
from tqdm import tqdm
from typing import Tuple, Union


def calculate_rarity(
        data: np.ndarray, 
        gamma: float = 0
    ) -> np.ndarray:
    """Calculate rarity for each column in data matrix based on the CoR condition.

    Parameters:
    ----------
        data: np.ndarray, shape = (n_samples, n_epochs).
        gamma: float, gamma < 1, hyper-parameter in the lower bound of coefficient of variation.
    
    Returns:
    -------
        rarity_list: np.ndarray, rarity of data's each column.
    """
    len_data = len(data)
    p_nc = np.arange(1, len_data) / len_data # nc -> non-critical
    p_c = 1 - p_nc # c -> critical
    data_sorted = np.sort(data, axis=0)
    data_sorted_cumsum = np.cumsum(data_sorted, axis=0)
    data_mean_nc = data_sorted_cumsum[:-1, :] / len_data
    data_mean_c = (data_sorted_cumsum[-1, :] - data_sorted_cumsum[:-1, :]) / len_data
    f_nc = data_mean_nc / p_nc[:, None]
    f_c = data_mean_c / p_c[:, None]
    condition = f_nc <= (p_c**(1-gamma/2))[:, None] * (f_c - f_nc)
    max_rarity_idx = np.sum(condition, axis=0) - 1
    rarity_list = -np.log10(p_c[max_rarity_idx])
    return rarity_list


def calculate_rarity_real(
        data_matrix: np.ndarray, 
        gamma: float = 0,
        debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the rarity given data as a vector of real numbers.
    
    Parameters:
    ----------
        data: np.ndarray, shape = (n_epochs, n_samples).
        gamma: float, gamma < 1, hyper-parameter in the lower bound of coefficient of variation.
        debug: bool, whether debug or not.
    
    Returns:
    -------
        rarity_list: np.ndarray, rarity of data's each column.
    """
    start_indices = np.zeros(len(data_matrix), dtype=int)
    rarity_list = np.zeros(len(data_matrix))
    max_rarity_indices = np.zeros(len(data_matrix), dtype=int)
    for i in tqdm(range(len(data_matrix))):
        data = data_matrix[i]
        if debug: print(f"data.sum() = {data.sum()}")
        if data.sum() < 0: data = -data
        len_data = len(data)
        data_sorted = np.sort(data)
        if debug: print(f"data_sorted.sum() = {data_sorted.sum()}, data_sorted = {data_sorted}")
        data_sorted_cumsum = np.cumsum(data_sorted)
        if debug: print(f"data_sorted_cumsum = {data_sorted_cumsum}")
        start_idx = len(data_sorted_cumsum[data_sorted_cumsum < 0])
        if debug: print(f"len_data ={len_data}, data_sorted_cumsum = {data_sorted_cumsum[-10:]}, start_idx = {start_idx}")
        start_indices[i] = start_idx
        p_nc = np.arange(start_idx+1, len_data) / len_data # nc -> non-critical
        p_c = 1 - p_nc # c -> critical
        data_mean_nc = data_sorted_cumsum[start_idx:-1] / len_data
        data_mean_c = (data_sorted_cumsum[-1] - data_sorted_cumsum[start_idx:-1]) / len_data
        f_nc = data_mean_nc / p_nc
        f_c = data_mean_c / p_c
        condition = f_nc <= (p_c**(1-gamma/2)) * (f_c - f_nc)
        max_rarity_idx = np.sum(condition) - 1
        max_rarity_indices[i] = max_rarity_idx
        rarity = -np.log10(p_c[max_rarity_idx])
        rarity_list[i] = rarity
    if debug: print(f"rarity = {rarity_list}")
    return start_indices, max_rarity_indices, rarity_list


def get_critical_samples(
        data: np.ndarray, 
        gamma: float = 0,
        return_rarity=False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Get critical samples' indices according to CoR theorem.

    Parameters:
    ----------
        data: np.ndarray, per-sample gradient dot products with mean gradient or gradient norms.
        gamma: float, gamma < 1, hyper-parameter in the lower bound of coefficient of variation.
    
    Returns:
    -------
        rarity: np.ndarray, rarity of data.
        critical_indices: indices of critical samples.
    """
    if data.sum() < 0: data = -data
    len_data = len(data)
    data_sorted = np.sort(data)
    data_sorted_cumsum = np.cumsum(data_sorted)
    start_idx = len(data_sorted_cumsum[data_sorted_cumsum < 0])
    p_nc = np.arange(start_idx+1, len_data) / len_data # nc -> non-critical
    p_c = 1 - p_nc # c -> critical
    data_mean_nc = data_sorted_cumsum[start_idx:-1] / len_data
    data_mean_c = (data_sorted_cumsum[-1] - data_sorted_cumsum[start_idx:-1]) / len_data
    f_nc = data_mean_nc / p_nc
    f_c = data_mean_c / p_c
    condition = f_nc <= (p_c**(1-gamma/2)) * (f_c - f_nc)
    max_rarity_idx = np.sum(condition) - 1
    rarity = -np.log10(p_c[max_rarity_idx])
    critical_indices = np.argsort(data)[start_idx+max_rarity_idx:]
    if return_rarity:
        return rarity, critical_indices
    else:
        return critical_indices


def cv_lower_bound(
        rarity_min: float, 
        rarity_max: float, 
        gamma: float = 0
    ) -> Tuple[float, float, float, float]:
    """Calculate lower bound of coefficient of variation (CV) given by the CoR theorem.

    Parameters:
    ----------
        rarity_min: float, min rarity of the data.
        rarity_max: float, max rarity of the data.
        gamma: float, hyper-parameter in the lower bound of coefficient of variation.
    
    Returns:
    -------
        y_min: float, min coefficient of variation.
        y_max: float, max coefficient of variation.
        slope: float, slope of the lower bound line.
        intercept: float, intercept of the lower bound line.
    """
    intercept = 0.5 * np.log10((1 - 10**(-rarity_min)) / 4)
    slope = 0.5 * min(1-gamma, 1)
    y_min = slope * rarity_min + intercept
    y_max = slope * rarity_max + intercept
    return y_min, y_max, slope, intercept


def cv_squared_lower_bound(
        rarity_min: float, 
        rarity_max: float, 
        gamma: float = 0
    ) -> Tuple[float, float, float, float]:
    """Calculate lower bound of coefficient of variation (CV) given by the CoR Theorem.

    Parameters:
    ----------
        rarity_min: float, min rarity of the data.
        rarity_max: float, max rarity of the data.
        gamma: float, hyper-parameter in the lower bound of coefficient of variation.
    
    Returns:
    -------
        y_min: float, min coefficient of variation.
        y_max: float, max coefficient of variation.
        slope: float, slope of the lower bound line.
        intercept: float, intercept of the lower bound line.
    """
    intercept = np.log10((1 - 10**(-rarity_min)) / 4)
    slope = min(1-gamma, 1)
    y_min = slope * rarity_min + intercept
    y_max = slope * rarity_max + intercept
    return y_min, y_max, slope, intercept


def num_samples_lower_bound(
        rarity_min: float, 
        rarity_max: float,
        const_ratio: float,
        gamma: float = 0
    ) -> Tuple[float, float, float, float]:
    """Calculate lower bound of required number of samples given by the CoR Theorem.

    Parameters:
    ----------
        rarity_min: float, min rarity of the data.
        rarity_max: float, max rarity of the data.
        const_ratio: float, z_alpha**2 / beta**2.
        gamma: float, hyper-parameter in the lower bound of coefficient of variation.
    
    Returns:
    -------
        y_min: float, min coefficient of variation.
        y_max: float, max coefficient of variation.
        slope: float, slope of the lower bound line.
        intercept: float, intercept of the lower bound line.
    """
    intercept = np.log10(const_ratio * (1 - 10**(-rarity_min)) / 4)
    slope = min(1-gamma, 1)
    y_min = slope * rarity_min + intercept
    y_max = slope * rarity_max + intercept
    return y_min, y_max, slope, intercept


def extend_line(two_points, x_min, x_max):
    x_1 = two_points[0, 0]
    x_2 = two_points[1, 0]
    y_1 = two_points[0, 1]
    y_2 = two_points[1, 1]
    slope = (y_2 - y_1) / (x_2 - x_1)
    intercept = y_1 - slope * x_1
    y_min = slope * x_min + intercept
    y_max = slope * x_max + intercept
    return y_min, y_max, slope, intercept
