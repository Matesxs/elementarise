import typing
import cv2
import numpy as np
import random
import multiprocessing
from functools import partial
import math
from tqdm import tqdm

from .metrics import get_total_metric, get_eval_metric
from .helpers import round_robin_generator, init_pool
from .utils import translate, draw_element, generate_output_image
from .param_generation import get_params
from .definitions import ElementType, TileSelectMode, string_to_tile_select_mode, string_to_element_type
from .metrics import get_window_metrics

def process_data(reference_image, output_image, element_type, max_size, min_size, min_width, max_width, min_height, max_height, min_alpha, max_alpha, _):
  element_type, params = get_params(element_type, min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)
  complete_params = [output_image, params, element_type]
  tmp_image, bounding_box = draw_element(*complete_params)
  prev_metric, new_metric = get_window_metrics(reference_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                               output_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                               tmp_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :])
  return prev_metric - new_metric, params, element_type

class Elementariser:
  def __init__(self, reference_image:np.ndarray, checkpoint_image:typing.Optional[np.ndarray]=None,
               process_scale_factor:float=1.0, output_scale_factor:float=1.0,
               num_of_elements:int=2000, batch_size:int=200, num_of_retries:int=20,
               width_divs:int=1, height_divs:int=1,
               min_alpha:int=1, max_alpha:int=255, max_size_start_coef:float=0.6, max_size_end_coef:float=0.1, max_size_decay_coef:float=1.0, min_size:int=2, element_type:typing.Union[ElementType, str]=ElementType.LINE,
               tile_select_mode:typing.Union[TileSelectMode, str]=TileSelectMode.RANDOM,
               workers:int=1, min_improvement:int=2000,
               save_progress:bool=False, progress_save_path:str="tmp",
               progress_callback:typing.Optional[typing.Callable[[np.ndarray, float], None]]=None,
               debug_on_progress_image:bool=False, debug:bool=False, use_tqdm:bool=False, visualise_progress:bool=False):
    assert process_scale_factor > 0, "Invalid process scale factor"
    assert output_scale_factor > 0, "Invalid output scale factor"
    assert width_divs >= 1 and height_divs >= 1, "Invalid image divisions"
    assert 1 <= min_alpha <= 255 and 1 <= max_alpha <= 255 and min_alpha <= max_alpha, "Invalid alpha settings"
    assert 0 < max_size_start_coef >= max_size_end_coef > 0 and min_size >= 1, "Invalid size settings"
    assert workers >= 1, "Invalid number of workers"
    assert min_improvement >= 0, "Invalid minimal improvement settings"

    if isinstance(element_type, str):
      element_type = string_to_element_type(element_type)
    if isinstance(tile_select_mode, str):
      tile_select_mode = string_to_tile_select_mode(tile_select_mode)

    self.progress_callback = progress_callback

    self.save_progress = save_progress
    self.progress_save_path = progress_save_path

    self.last_selected_zone_index = None
    self.all_zones = []

    original_width, original_height = reference_image.shape[1], reference_image.shape[0]
    if debug:
      print(f"Original size: {original_width}x{original_height}")
    self.reference_image = reference_image if process_scale_factor == 1 else cv2.resize(reference_image, dsize=None, fx=process_scale_factor, fy=process_scale_factor, interpolation=cv2.INTER_AREA)
    self.width, self.height = self.reference_image.shape[1], self.reference_image.shape[0]
    if debug:
      print(f"Processing size: {self.width}x{self.height}")

    self.process_image = None
    self.output_image = np.zeros((int(original_height * output_scale_factor), int(original_width * output_scale_factor), 3)) if checkpoint_image is None else checkpoint_image.copy()
    if debug:
      print(f"Output size: {self.output_image.shape[1]}x{self.output_image.shape[0]}")

    if checkpoint_image is not None:
      if checkpoint_image.shape != self.reference_image.shape:
        self.process_image = cv2.resize(checkpoint_image, dsize=(self.reference_image.shape[1], self.reference_image.shape[0]), interpolation=cv2.INTER_AREA)
      else:
        self.process_image = checkpoint_image
    else:
      self.process_image = np.zeros_like(self.reference_image)

    self.width_splits = width_divs
    self.height_splits = height_divs
    self.width_split_coef = math.ceil(self.width / width_divs)
    self.height_split_coef = math.ceil(self.height / height_divs)
    if debug:
      print(f"Window size: {self.width_split_coef}x{self.height_split_coef}")

    self.start_max_size = self.max_size = int(max(float(min_size), min(self.width_split_coef, self.height_split_coef) * max_size_start_coef))
    self.end_max_size = int(max(float(min_size), min(self.width_split_coef, self.height_split_coef) * max_size_end_coef))
    if debug:
      print(f"Start max size: {self.start_max_size}, End max size: {self.end_max_size}")

    self.min_size = min_size
    self.max_size_decay_coef = max_size_decay_coef
    self.min_alpha = min_alpha
    self.max_alpha = max_alpha

    self.elements = num_of_elements
    self.workers = workers
    self.batch_size = batch_size
    self.num_of_retries = num_of_retries
    self.tile_select_mode = tile_select_mode
    self.element_type = element_type
    self.min_improvement = min_improvement

    self.debug = debug
    self.debug_on_progress_image = debug_on_progress_image
    self.use_tqdm = use_tqdm
    self.visualise_progress = visualise_progress

    self.current_distance = get_total_metric(self.reference_image, self.process_image)

    for yidx in range(height_divs):
      min_height = self.height_split_coef * yidx
      max_height = min(self.height, self.height_split_coef * (yidx + 1))
      for xidx in range(width_divs):
        min_width = self.width_split_coef * xidx
        max_width = min(self.width, self.width_split_coef * (xidx + 1))
        self.all_zones.append((min_width, max_width, min_height, max_height))

    self.get_next_zone = round_robin_generator(self.all_zones.copy())

  def draw_splits(self, image, last_zone=None, last_bbox=None):
    for yidx in range(self.height_splits):
      y1 = self.height_split_coef * yidx
      y2 = min(self.height, self.height_split_coef * (yidx + 1))
      for xidx in range(self.width_splits):
        x1 = self.width_split_coef * xidx
        x2 = min(self.width, self.width_split_coef * (xidx + 1))
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color=((250, 50, 5) if (x1, x2, y1, y2) in self.all_zones else (15, 5, 245)))

    if last_zone is not None:
      cv2.rectangle(image, (last_zone[0], last_zone[2]), (last_zone[1] - 1, last_zone[3] - 1), color=(15, 200, 250))

    if last_bbox is not None:
      cv2.rectangle(image, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), color=(240, 190, 15))

  def get_zone_data(self):
    if len(self.all_zones) > 1:
      if self.tile_select_mode == TileSelectMode.PRIORITY:
        metrics = np.array([get_eval_metric(self.reference_image[y1:y2, x1:x2, :], self.process_image[y1:y2, x1:x2, :]) for x1, x2, y1, y2 in self.all_zones])
        selected_index = metrics.argmax()
        zone_data = self.all_zones[selected_index]
      elif self.tile_select_mode == TileSelectMode.ROUND_ROBIN:
        zone_data = self.get_next_zone()
        while zone_data not in self.all_zones:
          zone_data = self.get_next_zone()
      else:
        zone_data = random.choice(self.all_zones)
    else:
      zone_data = self.all_zones[0]

    return zone_data

  def call_callback(self, progress:int, last_zone=None, last_bbox=None):
    if self.progress_callback is not None or self.visualise_progress:
      prog_image = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2BGR)
      if self.debug_on_progress_image:
        self.draw_splits(prog_image, last_zone, last_bbox)

      if self.visualise_progress:
        cv2.imshow("progress_window", prog_image)
        cv2.setWindowTitle("progress_window", f"{progress}/{self.elements}")
        cv2.waitKey(1)

      if self.progress_callback is not None:
        self.progress_callback(cv2.cvtColor(prog_image, cv2.COLOR_BGR2RGB), progress / self.elements)

  def run(self) -> np.ndarray:
    param_store = []

    try:
      _indexes = list(range(self.batch_size))

      self.call_callback(0)

      iterator = range(self.elements) if not self.use_tqdm else tqdm(range(self.elements), unit="element")

      with multiprocessing.Pool(self.workers, init_pool) as executor:
        for iteration in iterator:
          self.max_size = int(max(float(self.min_size), translate(self.max_size_decay_coef * iteration, 0, self.elements, self.start_max_size, self.end_max_size)))

          zone_data = self.get_zone_data()

          min_width, max_width, min_height, max_height = zone_data
          get_params_function = partial(process_data, self.reference_image, self.process_image, self.element_type, self.max_size, self.min_size, min_width, max_width, min_height, max_height, self.min_alpha, self.max_alpha)

          retries = 0
          while True:
            scored_params = executor.map(get_params_function, _indexes)
            scored_params = [param for param in scored_params if param is not None]
            scored_params.sort(key=lambda x: x[0], reverse=True)

            distance_diff = scored_params[0][0]
            is_better = distance_diff > self.min_improvement

            if is_better:
              break
            else:
              retries += 1
              if retries >= self.num_of_retries:
                if len(self.all_zones) > 1:
                  retries = 0
                  self.all_zones.remove(zone_data)
                  zone_data = self.get_zone_data()

                  self.call_callback(iteration, zone_data)
                  continue
                break

          if retries >= self.num_of_retries:
            break

          params = scored_params[0][1]
          element_type = scored_params[0][2]
          param_store.append((element_type, params))

          params = [self.process_image, params, element_type]
          self.process_image, bbox = draw_element(*params)

          self.current_distance -= distance_diff

          self.call_callback(iteration, zone_data, bbox)
    except KeyboardInterrupt:
      if self.debug:
        print("Interrupted by user")
      pass

    self.call_callback(self.elements)
    if self.visualise_progress:
      cv2.destroyWindow("progress_window")

    return generate_output_image(self.output_image, param_store, self.output_image.shape[1] / self.width, self.save_progress, self.progress_save_path, use_tqdm=self.use_tqdm, debug=self.debug)
