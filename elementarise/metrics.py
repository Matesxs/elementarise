from numba import njit
import numpy as np
import math

@njit(nogil=True)
def get_process_metric(original, fake):
  return np.sum((original - fake) ** 2)

@njit(nogil=True)
def get_eval_metric(original, fake):
  mse = np.mean((original - fake) ** 2)
  if mse == 0: return np.inf
  return 20 * math.log10(255.0 / math.sqrt(mse))

@njit(nogil=True)
def get_window_metrics(original_window, process_window, current_window, metric_function):
  prev_distance = metric_function(original_window, process_window)
  new_distance = metric_function(original_window, current_window)
  return prev_distance, new_distance
