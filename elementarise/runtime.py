import typing
import cv2
import numpy as np
import random
import multiprocessing
from functools import partial
import math
from tqdm import tqdm

from .metrics import get_window_metrics, get_process_metric, get_eval_metric
from .helpers import round_robin_generator, init_pool
from .utils import translate, draw_element, generate_output_image
from .param_generation import get_params
from .definitions import ElementType, TileSelectMode, string_to_tile_select_mode, string_to_element_type, MetricsMode, string_to_metrics_mode

def process_data(reference_image:np.ndarray, process_image:np.ndarray, element_type:ElementType, max_size:int, min_size:int, min_width:int, max_width:int, min_height:int, max_height:int, min_alpha:int, max_alpha:int, process_metric_function:typing.Optional[typing.Callable[[np.ndarray, np.ndarray],float]], _) -> typing.Tuple[float, tuple, ElementType]:
  element_type, params = get_params(element_type, min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)
  complete_params = [process_image, params, element_type]
  tmp_image, bounding_box = draw_element(*complete_params)
  prev_metric, new_metric = get_window_metrics(reference_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                               process_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                               tmp_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                               process_metric_function)
  return prev_metric - new_metric, params, element_type

class Elementariser:
  def __init__(self,
               reference_image:np.ndarray,
               checkpoint_image:typing.Optional[np.ndarray]=None,
               process_scale_factor:float=1.0,
               output_scale_factor:float=1.0,
               num_of_elements:int=2000,
               batch_size:int=200,
               num_of_retries:int=20,
               width_divs:int=1,
               height_divs:int=1,
               min_alpha:int=1,
               max_alpha:int=255,
               max_size_start_coef:float=0.4,
               max_size_end_coef:float=0.1,
               max_size_decay_coef:float=1.0,
               min_size:int=2,
               element_type:typing.Union[ElementType, str]=ElementType.LINE,
               tile_select_mode:typing.Union[TileSelectMode, str]=TileSelectMode.RANDOM,
               tile_target:typing.Optional[typing.Tuple[int, int]]=None,
               workers:int=1,
               save_progress:bool=False,
               progress_save_path:str="tmp",
               progress_callback:typing.Optional[typing.Callable[[np.ndarray, float], None]]=None,
               custom_process_metrics:typing.Optional[typing.Tuple[typing.Callable[[np.ndarray, np.ndarray], float], typing.Union[MetricsMode, str]]]=None,
               custom_evaluation_metrics:typing.Optional[typing.Tuple[typing.Callable[[np.ndarray, np.ndarray], float], typing.Union[MetricsMode, str]]]=None,
               min_improvement: int=2000,
               debug_on_progress_image:bool=False,
               debug:bool=False,
               use_tqdm:bool=False,
               visualise_progress:bool=False):
    """
    Elementariser for reference image

    :param reference_image: Reference image
    :param checkpoint_image: Checkpoint image
    :param process_scale_factor: Scale of processing image
    :param output_scale_factor: Scale of output image
    :param num_of_elements: Number of elements to draw
    :param batch_size: Number of test elements in epoch
    :param num_of_retries: Limit number of tries per element
    :param width_divs: Number of divisions per width
    :param height_divs: Number of divisions per height
    :param min_alpha: Minimal alpha value of element
    :param max_alpha: Maxmimum alpha value of element
    :param max_size_start_coef: Maximum element size start coef
    :param max_size_end_coef: Maximum element size final coef
    :param max_size_decay_coef: Maximum element size decay coef
    :param min_size: Minimum element size
    :param element_type: Element used for recreating reference image
    :param tile_select_mode: Tile select mode changes behaviour of tile selection when multiple of them are present
    :param tile_target: Target tile for TARGET tile select mode
    :param workers: Number of workers for generating elements
    :param save_progress: Store progress of generation
    :param progress_save_path: Path to folder where progress imagis will be saved
    :param progress_callback: Function that will be called on each each progress step
    :param custom_process_metrics: Custom metrics function for generating image
    :param custom_evaluation_metrics: Custom metrics function for evaluation of progress image (used for priory mode)
    :param min_improvement: Minimal improvement of metrics for adding new element
    :param debug_on_progress_image: Draw progress details on progress image
    :param debug: Print debug
    :param use_tqdm: Show progress using tqdm
    :param visualise_progress: Show progress image using cv2
    """

    assert process_scale_factor > 0, "Invalid process scale factor, it should be larger than 0"
    assert output_scale_factor > 0, "Invalid output scale factor, it should be larger than 0"
    assert width_divs >= 1 and height_divs >= 1, "Invalid image divisions, divisions should be >= 1"
    assert 1 <= min_alpha <= 255 and 1 <= max_alpha <= 255 and min_alpha <= max_alpha, "Invalid alpha settings, alpha value can be in range 1 - 255 and min alpha need to be smaller or equal to max alpha"
    assert 0 < max_size_start_coef >= max_size_end_coef > 0 and min_size >= 1, "Invalid size settings"
    assert workers >= 1, "Invalid number of workers"
    assert min_improvement >= 0, "Invalid minimal improvement settings, improvement need to be >= 0"
    if tile_select_mode == TileSelectMode.TARGET:
      assert tile_target is not None, "Tile target not set"
      assert 0 <= tile_target[0] < width_divs and 0 <= tile_target[1] < height_divs, "Invalid tile target"

    if isinstance(element_type, str):
      element_type = string_to_element_type(element_type)
    if isinstance(tile_select_mode, str):
      tile_select_mode = string_to_tile_select_mode(tile_select_mode)

    self.process_metrics = get_process_metric
    self.process_metrics_mode = MetricsMode.MINIMALISE
    self.eval_metrics = get_eval_metric
    self.eval_metrics_mode = MetricsMode.MAXIMALISE
    if custom_process_metrics is not None:
      self.process_metrics = custom_process_metrics[0]
      self.process_metrics_mode = custom_process_metrics[1]
      if isinstance(self.process_metrics_mode, str):
        self.process_metrics_mode = string_to_metrics_mode(self.process_metrics_mode)
    if custom_evaluation_metrics is not None:
      self.eval_metrics = custom_evaluation_metrics[0]
      self.eval_metrics_mode = custom_evaluation_metrics[1]
      if isinstance(self.eval_metrics_mode, str):
        self.eval_metrics_mode = string_to_metrics_mode(self.eval_metrics_mode)

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
    self.tile_target = tile_target
    self.element_type = element_type
    self.min_improvement = min_improvement

    self.debug = debug
    self.debug_on_progress_image = debug_on_progress_image
    self.use_tqdm = use_tqdm
    self.visualise_progress = visualise_progress

    self.interrupted = False

    self.current_distance = self.process_metrics(self.reference_image, self.process_image)

    for yidx in range(height_divs):
      min_height = self.height_split_coef * yidx
      max_height = min(self.height, self.height_split_coef * (yidx + 1))
      for xidx in range(width_divs):
        min_width = self.width_split_coef * xidx
        max_width = min(self.width, self.width_split_coef * (xidx + 1))
        self.all_zones.append((min_width, max_width, min_height, max_height))

    self.get_next_zone = round_robin_generator(self.all_zones.copy())

  def __draw_splits(self, image, last_zone=None, last_bbox=None):
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

  def __get_zone_data(self):
    if len(self.all_zones) > 1:
      if self.tile_select_mode == TileSelectMode.PRIORITY:
        metrics = np.array([self.eval_metrics(self.reference_image[y1:y2, x1:x2, :], self.process_image[y1:y2, x1:x2, :]) for x1, x2, y1, y2 in self.all_zones])
        selected_index = metrics.argmax() if self.eval_metrics_mode == MetricsMode.MINIMALISE else metrics.argmin()
        zone_data = self.all_zones[selected_index]
      elif self.tile_select_mode == TileSelectMode.ROUND_ROBIN:
        zone_data = self.get_next_zone()
        while zone_data not in self.all_zones:
          zone_data = self.get_next_zone()
      elif self.tile_select_mode == TileSelectMode.ONE_BY_ONE:
        zone_data = self.all_zones[0]
      elif self.tile_select_mode == TileSelectMode.TARGET:
        zone_data = (self.width_split_coef * self.tile_target[0], self.width_split_coef * (self.tile_target[0] + 1), self.height_split_coef * self.tile_target[1], self.height_split_coef * (self.tile_target[1] + 1))
      else:
        zone_data = random.choice(self.all_zones)
    else:
      zone_data = self.all_zones[0]

    return zone_data

  def __call_callback(self, progress:int, last_zone=None, last_bbox=None):
    if self.progress_callback is not None or self.visualise_progress:
      prog_image = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2BGR)
      if self.debug_on_progress_image:
        self.__draw_splits(prog_image, last_zone, last_bbox)

      if self.visualise_progress:
        diff_image = np.abs((self.reference_image - self.process_image)).sum(axis=2) / 3
        diff_image = cv2.cvtColor(diff_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.__draw_splits(diff_image, last_zone, last_bbox)

        display_image = np.zeros((prog_image.shape[0], prog_image.shape[1] * 2, prog_image.shape[2]), dtype=np.uint8)
        display_image[:, :prog_image.shape[1], :] = prog_image
        display_image[:, prog_image.shape[1]:, :] = diff_image

        display_image_width, display_image_height = display_image.shape[1], display_image.shape[0]
        if display_image_width > 1920 or display_image_height > 1080:
          scaler1 = 1920 / display_image_width
          scaler2 = 1080 / display_image_height
          scaler = min(scaler1, scaler2)
          display_image = cv2.resize(display_image, dsize=None, fx=scaler, fy=scaler, interpolation=cv2.INTER_AREA)

        cv2.imshow("progress_window", display_image)
        cv2.setWindowTitle("progress_window", f"{progress}/{self.elements}, Size: {self.min_size}-{self.max_size}")
        cv2.waitKey(1)

      if self.progress_callback is not None:
        self.progress_callback(cv2.cvtColor(prog_image, cv2.COLOR_BGR2RGB), progress / self.elements)

  def run(self) -> typing.Tuple[np.ndarray, int]:
    """
    :return: Final image and elements drawn
    """

    param_store = []
    elements = 0

    try:
      _indexes = list(range(self.batch_size))

      self.__call_callback(0)

      iterator = range(self.elements) if not self.use_tqdm else tqdm(range(self.elements), unit="element")

      with multiprocessing.Pool(self.workers, init_pool) as executor:
        for iteration in iterator:
          self.max_size = int(max(float(self.end_max_size), translate(self.max_size_decay_coef * iteration, 0, self.elements, self.start_max_size, self.end_max_size)))

          zone_data = self.__get_zone_data()

          min_width, max_width, min_height, max_height = zone_data
          get_params_function = partial(process_data, self.reference_image, self.process_image, self.element_type, self.max_size, self.min_size, min_width, max_width, min_height, max_height, self.min_alpha, self.max_alpha, self.process_metrics)

          retries = 0
          while True:
            scored_params = executor.map(get_params_function, _indexes)
            scored_params = [param for param in scored_params if param is not None]
            scored_params.sort(key=lambda x: x[0], reverse=(True if self.process_metrics_mode == MetricsMode.MINIMALISE else False))

            distance_diff = scored_params[0][0]
            is_better = (distance_diff >= self.min_improvement) if self.process_metrics_mode == MetricsMode.MINIMALISE else (distance_diff <= self.min_improvement)

            if is_better:
              break
            else:
              retries += 1
              if retries >= self.num_of_retries:
                if len(self.all_zones) > 1 and self.tile_select_mode != TileSelectMode.TARGET:
                  retries = 0
                  self.all_zones.remove(zone_data)
                  zone_data = self.__get_zone_data()

                  min_width, max_width, min_height, max_height = zone_data
                  get_params_function = partial(process_data, self.reference_image, self.process_image, self.element_type, self.max_size, self.min_size, min_width, max_width, min_height, max_height, self.min_alpha, self.max_alpha, self.process_metrics)

                  self.__call_callback(iteration + 1, zone_data)
                  if isinstance(iterator, tqdm):
                    iterator.set_description(f"Size: {self.min_size}-{self.max_size}")
                  continue
                break

          if retries >= self.num_of_retries:
            break

          params = scored_params[0][1]
          element_type = scored_params[0][2]
          param_store.append((element_type, params))

          params = [self.process_image, params, element_type]
          self.process_image, bbox = draw_element(*params)
          elements += 1

          self.current_distance -= distance_diff

          self.__call_callback(iteration + 1, zone_data, bbox)
          if isinstance(iterator, tqdm):
            iterator.set_description(f"Size: {self.min_size}-{self.max_size}")
    except KeyboardInterrupt:
      self.interrupted = True
      if self.debug:
        print("Interrupted by user")
      pass

    self.__call_callback(self.elements)
    if self.visualise_progress:
      cv2.destroyWindow("progress_window")

    return generate_output_image(self.output_image, param_store, self.output_image.shape[1] / self.width, self.save_progress, self.progress_save_path, use_tqdm=self.use_tqdm, debug=self.debug), elements

class AutoElementariser:
  def __init__(self,
               reference_image: np.ndarray,
               checkpoint_image: typing.Optional[np.ndarray] = None,
               output_scale_factor: float = 1.0,
               num_of_elements: int = 5000,
               batch_size: int = 200,
               num_of_retries: int = 50,
               element_type:typing.Union[ElementType, str]=ElementType.LINE,
               workers: int = 1,
               save_progress: bool = False,
               progress_save_path: str = "tmp",
               progress_callback: typing.Optional[typing.Callable[[np.ndarray, float], None]] = None,
               custom_process_metrics: typing.Optional[typing.Tuple[typing.Callable[[np.ndarray, np.ndarray], float], typing.Union[MetricsMode, str]]] = None,
               custom_evaluation_metrics: typing.Optional[typing.Tuple[typing.Callable[[np.ndarray, np.ndarray], float], typing.Union[MetricsMode, str]]] = None,
               min_improvement: int = 2000,
               debug_on_progress_image: bool = False,
               debug: bool = False,
               use_tqdm: bool = False,
               visualise_progress: bool = False):
    """
    Prebuild elementariser with generation scheduler

    :param reference_image: Reference image
    :param checkpoint_image: Checkpoint image
    :param output_scale_factor: Scale of output image
    :param num_of_elements: Number of elements to draw
    :param batch_size: Number of test elements in epoch
    :param num_of_retries: Limit number of tries per element
    :param element_type: Element used for recreating reference image
    :param workers: Number of workers for generating elements
    :param save_progress: Store progress of generation
    :param progress_save_path: Path to folder where progress imagis will be saved
    :param progress_callback: Function that will be called on each each progress step
    :param custom_process_metrics: Custom metrics function for generating image
    :param custom_evaluation_metrics: Custom metrics function for evaluation of progress image (used for priory mode)
    :param min_improvement: Minimal improvement of metrics for adding new element
    :param debug_on_progress_image: Draw progress details on progress image
    :param debug: Print debug
    :param use_tqdm: Show progress using tqdm
    :param visualise_progress: Show progress image using cv2
    """

    assert output_scale_factor > 0, "Invalid output scale factor, it should be larger than 0"
    assert workers >= 1, "Invalid number of workers"
    assert min_improvement >= 0, "Invalid minimal improvement settings, improvement need to be >= 0"

    if isinstance(element_type, str):
      element_type = string_to_element_type(element_type)

    self.reference_image = reference_image
    self.checkpoint_image = checkpoint_image

    self.output_scale_factor = output_scale_factor

    self.num_of_elements = num_of_elements
    self.batch_size = batch_size
    self.num_of_retries = num_of_retries
    self.element_type = element_type

    self.workers = workers

    self.save_progress = save_progress
    self.progress_save_path = progress_save_path

    self.progress_callback = progress_callback

    self.custom_progress_metrics = custom_process_metrics
    self.custom_evaluation_metrics = custom_evaluation_metrics
    self.min_improvement = min_improvement

    self.debug_on_progress_image = debug_on_progress_image
    self.debug = debug
    self.use_tqdm = use_tqdm
    self.visualise_progress = visualise_progress

  def run(self) -> typing.Tuple[np.ndarray, int]:
    """
    :return: Final image and elements drawn
    """

    all_generated_elements = 0

    for scale, divs in zip([0.125, 0.25, 0.5, 1], [2, 3, 6, 12]):
      elementariser = Elementariser(self.reference_image,
                                    self.checkpoint_image,
                                    scale,
                                    self.output_scale_factor,
                                    self.num_of_elements,
                                    self.batch_size,
                                    self.num_of_retries,
                                    divs, divs,
                                    element_type=self.element_type,
                                    workers=self.workers,
                                    save_progress=self.save_progress,
                                    progress_save_path=self.progress_save_path,
                                    progress_callback=self.progress_callback,
                                    custom_process_metrics=self.custom_progress_metrics,
                                    custom_evaluation_metrics=self.custom_evaluation_metrics,
                                    min_improvement=self.min_improvement,
                                    debug_on_progress_image=self.debug_on_progress_image,
                                    debug=self.debug,
                                    use_tqdm=self.use_tqdm,
                                    visualise_progress=self.visualise_progress)
      self.checkpoint_image, generated_elements = elementariser.run()
      all_generated_elements += generated_elements
      self.num_of_elements -= generated_elements
      if self.num_of_elements <= 0 or elementariser.interrupted:
        break

    return self.checkpoint_image, all_generated_elements