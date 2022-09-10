import numpy as np
from PIL import Image
from numba import njit
import signal
from itertools import cycle

def calculate_polygon_coords(x_center, y_center, radius, starting_angle, number_of_vertices):
  angle_increment = np.pi * 2 / number_of_vertices
  return np.array([(int(x_center + np.cos(starting_angle + i * angle_increment) * radius),
                    int(y_center + np.sin(starting_angle + i * angle_increment) * radius)) for i in range(number_of_vertices)], dtype=int).flatten()

def numpy_to_image(array, input_format="RGB", output_format="RGBA"):
  return Image.fromarray(array, mode=input_format).convert(mode=output_format)

def image_to_numpy(image, output_mode="RGB"):
  return np.array(image.convert(mode=output_mode))

@njit(nogil=True)
def get_line_boundingbox(xy, thickness, width, height):
  points = xy.reshape((2, 2))
  vector = points[1, :] - points[0, :]
  norm = vector / np.linalg.norm(vector)

  perpendicular = np.array([-norm[1], norm[0]]) * np.ceil(thickness * 0.5)
  perpendicular = perpendicular.reshape((1, -1))

  offsets = np.concatenate((perpendicular, perpendicular), axis=1).reshape((-1, 2)) * np.array([[1], [-1]])

  abcd = np.concatenate((points, points), axis=1).reshape((-1, 2)) + np.concatenate((offsets, offsets), axis=0)
  xs = abcd[:, 0]
  ys = abcd[:, 1]
  return np.array([max(0, xs.min() - 1), max(0, ys.min() - 1), min(width, xs.max() + 1), min(height, ys.max() + 1)])

def init_pool():
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def round_robin_generator(item_list):
  iterator = cycle(item_list)

  def get_next():
    return next(iterator)

  return get_next
