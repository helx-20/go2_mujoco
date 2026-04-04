import numpy as np
import os
from scipy.stats import norm
import math

alpha = 0.05
z = norm.isf(q=alpha)

def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var

def analyze(path):
    crashes = []
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file), allow_pickle=True).tolist()
        crashes.extend(data)
    mean, rhf, var = calculate_val(crashes)
    print(f'Failure rate: {np.sum(crashes) / len(crashes)}')
    print(f'Mean: {mean[-1]:.6f}, Relative Half Width: {rhf[-1]:.6f}, Variance: {var[-1]:.6f}')
    print(f'Total samples: {len(crashes)}, Crashes: {np.sum(crashes)}')

if __name__ == '__main__':
    root = 'results/nde'
    analyze(root)