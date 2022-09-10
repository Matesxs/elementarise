import random
import numpy as np
from numba import njit
import typing

from .definitions import ElementType

@njit(nogil=True)
def get_random_color(min_alpha, max_alpha):
  return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(min_alpha, max_alpha)

@njit(nogil=True)
def get_line_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha):
  return random.randint(min_width, max_width - 1), \
         random.randint(min_height, max_height - 1), \
         random.randint(min_size, max_size), \
         random.randint(min_size, max_size), \
         random.random() * np.pi * 2, \
         get_random_color(min_alpha, max_alpha)

@njit(nogil=True)
def get_circle_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha):
  return random.randint(min_width, max_width - 1), \
         random.randint(min_height, max_height - 1), \
         random.randint(min_size, max_size), \
         get_random_color(min_alpha, max_alpha)

@njit(nogil=True)
def get_symmetrical_polygon_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha):
  radius = random.randint(min_size, max_size)
  x_center = random.randint(min_width, max_width - 1)
  y_center = random.randint(min_height, max_height - 1)
  starting_angle = random.random() * np.pi * 2
  return x_center, \
         y_center, \
         radius, \
         starting_angle, \
         get_random_color(min_alpha, max_alpha)

__mode_switcher = {
  ElementType.LINE: get_line_params,
  ElementType.CIRCLE: get_circle_params,
  ElementType.TRIANGLE: get_symmetrical_polygon_params,
  ElementType.SQUARE: get_symmetrical_polygon_params,
  ElementType.PENTAGON: get_symmetrical_polygon_params,
  ElementType.HEXAGON: get_symmetrical_polygon_params,
  ElementType.OCTAGON: get_symmetrical_polygon_params
}

def __get_random_params(min_width: int, max_width: int, min_height: int, max_height: int, max_size: int, min_size: int, min_alpha: int, max_alpha: int) -> typing.Tuple[ElementType, tuple]:
  element_type, get_par_func = random.choice(list(__mode_switcher.items()))
  return element_type, get_par_func(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)

def get_params(element_type: ElementType, min_width: int, max_width: int, min_height: int, max_height: int, max_size: int, min_size: int, min_alpha: int, max_alpha: int) -> typing.Tuple[ElementType, tuple]:
  if element_type == ElementType.RANDOM:
    return __get_random_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)

  assert element_type in __mode_switcher.keys(), "Invalid mode"
  get_par_func = __mode_switcher[element_type]

  return element_type, get_par_func(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)
