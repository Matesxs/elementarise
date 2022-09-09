from numba import njit
import numpy as np

@njit(nogil=True)
def get_total_metric(original, fake):
  return np.sum((original - fake) ** 2)

@njit(nogil=True)
def get_eval_metric(original, fake):
  return np.mean((original - fake) ** 2)

@njit(nogil=True)
def get_window_metrics(original_window, output_window, current_window):
  prev_distance = get_total_metric(original_window, output_window)
  new_distance = get_total_metric(original_window, current_window)
  return prev_distance, new_distance
