import threading
import multiprocessing
import ctypes
import queue
import typing
import numpy as np
from PIL import Image
import time
import math
import os
import pathlib
from numba import njit
from tqdm import tqdm

from .drawing import draw_element
from .definitions import ElementType

@njit(nogil=True)
def translate(value:typing.Union[int, float], leftMin:typing.Union[int, float], leftMax:typing.Union[int, float], rightMin:typing.Union[int, float], rightMax:typing.Union[int, float]) -> float:
  leftSpan = leftMax - leftMin
  rightSpan = rightMax - rightMin
  valueScaled = float(value - leftMin) / float(leftSpan)
  return rightMin + (valueScaled * rightSpan)

def save_image(data:typing.Tuple[np.ndarray, str]):
  image, path = data
  image = Image.fromarray(image, mode="RGB")
  image.save(path)

class ImageSaver(threading.Thread):
  def __init__(self):
    super(ImageSaver, self).__init__(daemon=True)

    self.stopped = multiprocessing.Value(ctypes.c_bool, False)
    self.data_queue = multiprocessing.Queue(maxsize=200)

  def put_data(self, image:np.ndarray, path:str):
    self.data_queue.put((image, path), block=True)

  def end(self):
    time.sleep(1)
    self.stopped.value = True

  def get_data(self, limit=None):
    data = []
    retrieved = 0
    start_time = time.time()
    while True:
      try:
        data.append(self.data_queue.get(block=True, timeout=10))
        retrieved += 1
        if limit is not None:
          if retrieved >= limit or time.time() - start_time > 20:
            break
      except queue.Empty:
        break
    return data

  def run(self) -> None:
    with multiprocessing.Pool(math.ceil(multiprocessing.cpu_count() / 2)) as executor:
      while not self.stopped.value:
        data = self.get_data(limit=200)
        if not data:
          continue

        executor.map(save_image, data)

      data = self.get_data()
      if data:
        executor.map(save_image, data)

def generate_output_image(output_image:np.ndarray, params_history:list, scale_factor:float=1.0, save_progress:bool=False, save_progress_path:str="tmp", use_tqdm:bool=False, debug: bool=False) -> np.ndarray:
  index_offset = 0
  image_saver = None
  if save_progress:
    image_saver = ImageSaver()
    image_saver.start()

    if not os.path.exists(save_progress_path):
      os.mkdir(save_progress_path)
    else:
      files = [pathlib.Path(p).stem for p in os.listdir(save_progress_path)]
      if files:
        indexes = [int(f) for f in files if f.isnumeric()]
        index_offset = max(indexes) + 1

  if debug:
    print("Starting final image generation")

  iterator = enumerate(params_history) if not use_tqdm else tqdm(enumerate(params_history), unit="image", total=len(params_history))

  try:
    for idx, param in iterator:
      element_type = param[0]
      element_params = param[1]

      if element_type == ElementType.LINE:
        params_to_scale = 4
      elif element_type == ElementType.CIRCLE or \
           element_type == ElementType.TRIANGLE or \
           element_type == ElementType.SQUARE or \
           element_type == ElementType.PENTAGON or \
           element_type == ElementType.HEXAGON or \
           element_type == ElementType.OCTAGON:
        params_to_scale = 3
      else:
        raise Exception("Invalid element type")

      params = [output_image,
                [*[int(p * scale_factor) for p in element_params[:params_to_scale]], *[p for p in element_params[params_to_scale:]]],
                element_type]
      output_image, _ = draw_element(*params)

      if save_progress:
        image_saver.put_data(output_image.copy(), f"{save_progress_path}/{idx + index_offset}.png")
  except KeyboardInterrupt:
    pass

  if image_saver is not None and image_saver.is_alive():
    if debug:
      print("Waiting for image saver to finish")
    image_saver.end()
    image_saver.join()

  return output_image
