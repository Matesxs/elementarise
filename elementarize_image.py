import argparse
import math
import os
from PIL import Image, ImageDraw
import numpy as np
from numba import njit
import random
import cv2
import time
from tqdm import tqdm
from functools import partial
import shutil
import subprocess
import pathlib
import threading
import multiprocessing
import ctypes
import queue
import signal
from itertools import cycle

MINIMALISE_EVAL_METRICS = True
MINIMALISE_DISTANCE = True
MIN_DIFF = 5_000

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
def get_ellipse_params(min_width, max_width, min_height, max_height, min_alpha, max_alpha):
  x_coors = np.random.randint(min_width, max_width, size=2)
  x_coors.sort()
  y_coors = np.random.randint(min_height, max_height, size=2)
  y_coors.sort()
  return x_coors[0], \
         y_coors[0], \
         x_coors[1], \
         y_coors[1], \
         get_random_color(min_alpha, max_alpha)

def calculate_coords(x_center, y_center, radius, starting_angle, number_of_vertices):
  angle_increment = np.pi * 2 / number_of_vertices
  return np.array([(int(x_center + np.cos(starting_angle + i * angle_increment) * radius),
                    int(y_center + np.sin(starting_angle + i * angle_increment) * radius)) for i in range(number_of_vertices)], dtype=int).flatten()

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

def get_random_element_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha, mode):
  if mode == 0:
    return get_line_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)
  elif mode == 1:
    return get_circle_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)
  elif mode == 2:
    return get_ellipse_params(min_width, max_width, min_height, max_height, min_alpha, max_alpha)
  else:
    return get_symmetrical_polygon_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha)

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

def draw_element(params, output_image, mode):
  height, width, _ = output_image.shape

  img = Image.fromarray(output_image, mode="RGB").convert(mode="RGBA")

  if mode == 0:
    overlay = Image.new("RGBA", img.size, params[5][:3] + (0,))
    draw = ImageDraw.Draw(overlay)
    end_point = (params[0] + np.cos(params[4]) * params[2], params[1] + np.sin(params[4]) * params[2])
    coords = [params[0], params[1], *end_point]
    draw.line(coords, params[5], params[3])
    bbox = get_line_boundingbox(np.array(coords, dtype=np.float64), params[3], width, height).astype(int)
  elif mode == 1 or mode == 2:
    if mode == 1:
      thickness = (params[2] - 1) / 2
      color = params[3]
      ellipse_params = (params[0] - thickness, params[1] - thickness, params[0] + thickness, params[1] + thickness)
    else:
      color = params[4]
      ellipse_params = (params[0], params[1], params[2], params[3])

    overlay = Image.new("RGBA", img.size, color[:3] + (0,))
    draw = ImageDraw.Draw(overlay)

    draw.ellipse(ellipse_params, fill=color)
    bbox = np.array([max(0, ellipse_params[0] - 1), max(0, ellipse_params[1] - 1), min(width, ellipse_params[2] + 1), min(height, ellipse_params[3] + 1)], dtype=int)
  else:
    color = params[4]
    overlay = Image.new("RGBA", img.size, color[:3] + (0,))
    draw = ImageDraw.Draw(overlay)

    coords = calculate_coords(params[0], params[1], params[2], params[3], mode)

    x_coords = []
    y_coords = []
    for idx in range(mode * 2):
      if idx % 2 == 0:
        x_coords.append(coords[idx])
      else:
        y_coords.append(coords[idx])

    draw.polygon([(xc, yc) for xc, yc in zip(x_coords, y_coords)], fill=color)
    bbox = np.array([max(0, min(x_coords) - 1), max(0, min(y_coords) - 1), min(width, max(x_coords) + 1), min(height, max(y_coords) + 1)], dtype=int)

  img.alpha_composite(overlay)

  return np.array(img.convert(mode="RGB")), bbox

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

def get_params(max_size, min_size, original_image, output_image, mode, min_width, max_width, min_height, max_height, min_alpha, max_alpha, _):
  params = get_random_element_params(min_width, max_width, min_height, max_height, max_size, min_size, min_alpha, max_alpha, mode)
  complete_params = [params, output_image, mode]
  tmp_image, bounding_box = draw_element(*complete_params)
  prev_metric, new_metric = get_window_metrics(original_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                                   output_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                                   tmp_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :])
  return prev_metric - new_metric, params

@njit(nogil=True)
def translate(value, leftMin, leftMax, rightMin, rightMax):
  leftSpan = leftMax - leftMin
  rightSpan = rightMax - rightMin
  valueScaled = float(value - leftMin) / float(leftSpan)
  return rightMin + (valueScaled * rightSpan)

def save_image(data):
  image, path = data
  image = Image.fromarray(image, mode="RGB")
  image.save(path)

class ImageSaver(threading.Thread):
  def __init__(self):
    super(ImageSaver, self).__init__(daemon=True)

    self.stopped = multiprocessing.Value(ctypes.c_bool, False)
    self.data_queue = multiprocessing.Queue(maxsize=200)

  def put_data(self, image, path):
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

def generate_output_image(output_image, params_history, scale_factor, save_progress_path):
  index_offset = 0
  image_saver = None
  if save_progress_path is not None:
    image_saver = ImageSaver()
    image_saver.start()

    if not os.path.exists(save_progress_path):
      os.mkdir(save_progress_path)
    else:
      files = [pathlib.Path(p).stem for p in os.listdir(save_progress_path)]
      if files:
        indexes = [int(f) for f in files if f.isnumeric()]
        index_offset = max(indexes) + 1

  print("Generating final image")
  for idx, param in tqdm(enumerate(params_history), total=len(params_history)):
    mode = param[0]
    element_params = param[1]

    if mode == 0:
      params_to_scale = 4
    elif mode == 1:
      params_to_scale = 3
    elif mode == 2:
      params_to_scale = 4
    else:
      params_to_scale = 3

    params = [[*[int(p * scale_factor) for p in element_params[:params_to_scale]], *[p for p in element_params[params_to_scale:]]],
              output_image,
              mode]
    output_image, _ = draw_element(*params)

    if save_progress_path is not None:
      image_saver.put_data(output_image.copy(), f"{save_progress_path}/{idx + index_offset}.png")

  if image_saver is not None and image_saver.is_alive():
    print("Waiting for image saver to finish")
    image_saver.end()
    image_saver.join()

  return Image.fromarray(output_image, mode="RGB")

def init_pool():
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def round_robin_generator(item_list):
  iterator = cycle(item_list)

  def get_next():
    return next(iterator)

  return get_next

class Elementariser:
  def __init__(self, width_divs, height_divs, width_coef, height_coef, width, height, min_alpha, max_alpha, max_size, max_size_minimum, max_size_decay_coef, min_size, reference_image, process_image, elements, workers, repeats, tries, tile_select_mode, mode):
    self.param_store = []
    self.last_selected_zone_index = None
    self.all_zones = []
    self.reference_image = reference_image
    self.process_image = process_image

    self.width_splits = width_divs
    self.height_splits = height_divs

    self.max_size = self.default_max_size = max_size
    self.max_size_minimum = max_size_minimum
    self.max_size_decay_coef = max_size_decay_coef
    self.min_size = min_size
    self.min_alpha = min_alpha
    self.max_alpha = max_alpha

    self.elements = elements
    self.workers = workers
    self.repeats = repeats
    self.tries = tries
    self.tile_select_mode = tile_select_mode
    self.mode = mode

    self.width = width
    self.height = height
    self.width_coef = width_coef
    self.height_coef = height_coef

    self.current_distance = get_total_metric(self.reference_image, self.process_image)

    for yidx in range(height_divs):
      min_height = height_coef * yidx
      max_height = min(height, height_coef * (yidx + 1))
      for xidx in range(width_divs):
        min_width = width_coef * xidx
        max_width = min(width, width_coef * (xidx + 1))
        self.all_zones.append((min_width, max_width, min_height, max_height))

    self.get_next_zone = round_robin_generator(self.all_zones.copy())

  def get_history(self):
    return self.param_store

  def draw_splits(self, image, last_zone=None, last_bbox=None):
    for yidx in range(self.height_splits):
      y1 = self.height_coef * yidx
      y2 = min(self.height, self.height_coef * (yidx + 1))
      for xidx in range(self.width_splits):
        x1 = self.width_coef * xidx
        x2 = min(self.width, self.width_coef * (xidx + 1))
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color=((250, 50, 5) if (x1, x2, y1, y2) in self.all_zones else (15, 5, 245)))

    if last_zone is not None:
      cv2.rectangle(image, (last_zone[0], last_zone[2]), (last_zone[1] - 1, last_zone[3] - 1), color=(15, 200, 250))

    if last_bbox is not None:
      cv2.rectangle(image, (last_bbox[0], last_bbox[1]), (last_bbox[2], last_bbox[3]), color=(240, 190, 15))

  def get_zone_data(self):
    if len(self.all_zones) > 1:
      if self.tile_select_mode == 2:
        metrics = np.array([get_eval_metric(self.reference_image[y1:y2, x1:x2, :], self.process_image[y1:y2, x1:x2, :]) for x1, x2, y1, y2 in self.all_zones])
        selected_index = metrics.argmax() if MINIMALISE_EVAL_METRICS else metrics.argmin()
        zone_data = self.all_zones[selected_index]
      elif self.tile_select_mode == 1:
        zone_data = self.get_next_zone()
        while zone_data not in self.all_zones:
          zone_data = self.get_next_zone()
      else:
        zone_data = random.choice(self.all_zones)
    else:
      zone_data = self.all_zones[0]

    return zone_data

  def run(self):
    try:
      _indexes = list(range(self.tries))
      iterator = tqdm(range(self.elements))

      prog_image = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2BGR)
      self.draw_splits(prog_image)
      cv2.imshow("progress_window", prog_image)

      cv2.setWindowTitle("progress_window", f"Progress: 0/{self.elements}, Distance: {self.current_distance}, Max size: {self.max_size}")

      iterator.set_description(f"Distance: {self.current_distance}, Max size: {self.max_size}")
      cv2.waitKey(1)

      with multiprocessing.Pool(self.workers, init_pool) as executor:
        for iteration in iterator:
          if self.max_size_minimum is not None:
            self.max_size = int(max(self.max_size_minimum, translate(self.max_size_decay_coef * iteration, 0, self.elements, self.default_max_size, self.max_size_minimum)))

          zone_data = self.get_zone_data()

          min_width, max_width, min_height, max_height = zone_data
          mode = self.mode if self.mode is not None else random.randint(0, 5)
          get_params_function = partial(get_params, self.max_size, self.min_size, self.reference_image, self.process_image, mode, min_width, max_width, min_height, max_height, self.min_alpha, self.max_alpha)

          repeats = 0
          while True:
            scored_params = list(executor.map(get_params_function, _indexes))
            scored_params = [param for param in scored_params if param is not None]
            scored_params.sort(key=lambda x: x[0], reverse=MINIMALISE_DISTANCE)

            distance_diff = scored_params[0][0]
            is_better = (distance_diff > MIN_DIFF) if MINIMALISE_DISTANCE else (distance_diff < MIN_DIFF)

            if is_better:
              break
            else:
              repeats += 1
              if repeats >= self.repeats:
                if len(self.all_zones) > 1:
                  repeats = 0
                  self.all_zones.remove(zone_data)
                  zone_data = self.get_zone_data()

                  prog_image = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2BGR)
                  self.draw_splits(prog_image)
                  cv2.imshow("progress_window", prog_image)
                  cv2.waitKey(1)
                  continue
                break

          if repeats >= self.repeats:
            print("Maximum retries reached in all zones")
            break

          params = scored_params[0][1]
          self.param_store.append((mode, params))

          params = [params, self.process_image, mode]
          self.process_image, bbox = draw_element(*params)

          prog_image = cv2.cvtColor(self.process_image, cv2.COLOR_RGB2BGR)
          self.draw_splits(prog_image, zone_data, bbox)

          self.current_distance -= distance_diff

          cv2.imshow("progress_window", prog_image)
          cv2.setWindowTitle("progress_window", f"Progress: {iteration + 1}/{self.elements}, Metric: {self.current_distance}, Diff: {distance_diff}, Max size: {self.max_size}")

          iterator.set_description(f"Metric: {self.current_distance}, Diff: {distance_diff}, Max size: {self.max_size}")
          cv2.waitKey(1)
    except KeyboardInterrupt:
      print("Interrupted by user")

    cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", "-i", help="Path to input image", type=str, required=True)
  parser.add_argument("--elements", "-e", help="Number of elements to draw", type=int, default=1000)
  parser.add_argument("--tries", "-t", help="Number of tries for each element", type=int, default=500)
  parser.add_argument("--repeats_limit", "-rl", help="Limit number of repeats per element", type=int, default=20)
  parser.add_argument("--tile_select_mode", "-tsm", help="Tile select mode changes behaviour of tile selection when multiple of them are present (default: 1)\n0 - random - tiles are selected randomly\n1 - round robin - tiles are selected one after another\n2 - worse first - tiles with worst metrics will get processed first", type=int, default=1)
  parser.add_argument("--size_multiplier", "-sm", help="Multiplier of size in connection to image (split) dimensions (size dont apply to mode 2)", type=float, default=0.4)
  parser.add_argument("--max_size_minimum", "-maxsm", help="Element size to which will maximum size decay overtime (if not set no decay will happen) (size dont apply to mode 2)", type=int, required=False)
  parser.add_argument("--min_size", "-mins", help="Minimum element size (size dont apply to mode 2)", type=int, default=2)
  parser.add_argument("--size_decay_coef", "-sdc", help="Coefficient of size decay", type=float, default=1)
  parser.add_argument("--min_alpha", "-mina", help="Minimal alpha value of element (default 1) (can't be less than 1)", type=int, default=1)
  parser.add_argument("--max_alpha", "-maxa", help="Maximum alpha value of element (default 255) (can't be more than 255)", type=int, default=255)
  parser.add_argument("--mode", "-m", help="Select element which will be generated (0 - line/rectangle, 1 - circle, 2 - ellipse, 3 - triangles, 4 - squares, 5 - pentagon, bigger mode values will generate coresponding polygon)", type=int, default=0)
  parser.add_argument("--random_mode", "-rm", action="store_true", help="Mode specifier gets ignored and mode is selected randomly for each element generation (modes 0 - 5 are used)")
  parser.add_argument("--width_splits", "-ws", help="Number of width splits for generating elements in smaller more specific areas (1 = no splits - default)", type=int, default=1)
  parser.add_argument("--height_splits", "-hs", help="Same as width splits only for height", type=int, default=1)
  parser.add_argument("--workers", "-w", help="Number of workers", type=int, default=2)
  parser.add_argument("--output", "-o", help="Path where to save output image", type=str, required=False)
  parser.add_argument("--progress_output", "-po", help="Path to folder where progress images will be saved", type=str, required=False)
  parser.add_argument("--progress_video", "-pv", help="Path to video of progress (works only with ffmpeg installed and in PATH) (if progress_output is not defined temporary folder with images will be created)", type=str, required=False)
  parser.add_argument("--progress_video_length", "-pvl", help="Approximate output progress video length in seconds (default 60)", type=int, default=60)
  parser.add_argument("--process_scale_factor", "-psf", help="Scale down factor for generating image (example: 2 will scale image size in both axis by factor of 2)", type=float, default=1)
  parser.add_argument("--output_scale_factor", "-osf", help="Scale factor for output image (same behaviour as process_scale_factor)", type=float, default=1)
  parser.add_argument("--checkpoint", "-ch", help="Checkpoint image path", type=str, required=False)

  args = parser.parse_args()

  assert os.path.exists(args.input) and os.path.isfile(args.input), "Invalid input file"
  assert args.elements > 0 and args.tries > 0 and args.repeats_limit > 0 and args.size_multiplier > 0, "Invalid image generation params"
  assert args.workers > 0, "Invalid number of workers"
  assert args.process_scale_factor > 0, "Invalid process scale factor"
  assert args.output_scale_factor > 0, "Invalid output scale factor"
  assert args.mode >= 0, "Invalid mode selected"
  if args.checkpoint is not None:
    assert os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint), "Invalid checkpoint file"
  assert args.width_splits >= 1 and args.height_splits >= 1, "Invalid split values"
  assert 1 <= args.min_alpha <= args.max_alpha <= 255, "Invalid element alpha settings"
  assert args.progress_video_length >= 1, "Invalid progress video length"
  assert args.size_decay_coef > 0, "Invalid size decay coefficient"
  assert args.min_size >= 1, "Invalid min size"
  assert 2 >= args.tile_select_mode >= 0, "Invalid tile select mode"
  if args.workers > multiprocessing.cpu_count():
    print(f"Number of processes to spawn ({args.workers}) is larger than number of processor cores ({multiprocessing.cpu_count()})!")

  input_image = Image.open(args.input).convert('RGB')
  # HxWxCH
  input_image = np.array(input_image)
  print(f"Original input image shape: {input_image.shape}")
  original_height, original_width, _ = input_image.shape
  output_image = np.zeros((int(original_height * args.output_scale_factor), int(original_width * args.output_scale_factor), 3))
  print(f"Expected output image shape: {output_image.shape}")

  if args.process_scale_factor != 1:
    resized_input_image = cv2.resize(input_image, dsize=None, fx=args.process_scale_factor, fy=args.process_scale_factor, interpolation=cv2.INTER_AREA)
  else:
    resized_input_image = input_image.copy()
  print(f"Processed input image shape: {resized_input_image.shape}")

  resized_height, resized_width, _ = resized_input_image.shape

  process_image = np.zeros_like(resized_input_image)
  if args.checkpoint is not None:
    checkpoint_image = Image.open(args.checkpoint).convert("RGB")
    checkpoint_image = np.array(checkpoint_image)
    process_image = cv2.resize(checkpoint_image, dsize=(process_image.shape[1], process_image.shape[0]), interpolation=cv2.INTER_AREA)
    output_image = checkpoint_image.copy() # We don't resize because we expect that this is wanted size and resizing will only introduce artifacts and image distortion

  width_split_coef = math.ceil(resized_width / args.width_splits)
  height_split_coef = math.ceil(resized_height / args.height_splits)
  max_size = min(width_split_coef, height_split_coef) * args.size_multiplier
  max_size = int(max(1, max_size))
  print(f"Tile size: {width_split_coef}x{height_split_coef}")

  if args.max_size_minimum is not None:
    assert 1 <= args.max_size_minimum <= max_size, f"Invalid minimum of max size, maximum is {max_size} for current settings"
    if args.max_size_minimum < args.min_size:
      args.min_size = args.max_size_minimum

  start_time = time.time()
  worker_manager = Elementariser(args.width_splits, args.height_splits, width_split_coef, height_split_coef, resized_width, resized_height, args.min_alpha, args.max_alpha, max_size, args.max_size_minimum, args.size_decay_coef, args.min_size, resized_input_image, process_image, args.elements, args.workers, args.repeats_limit, args.tries, args.tile_select_mode, args.mode if not args.random_mode else None)
  worker_manager.run()

  line_params_history = worker_manager.get_history()

  print(f"{len(line_params_history)} elements, Elapsed: {(time.time() - start_time) / 60}mins")

  progress_images_path = args.progress_output if args.progress_output is not None else ("tmp" if args.progress_video is not None else None)
  output_image_object = generate_output_image(output_image, line_params_history, output_image.shape[1] / resized_width, progress_images_path)

  if args.output is not None:
    output_image_object.save(args.output)

  if args.progress_video is not None:
    print("Generating progress video")
    number_of_frames = len(os.listdir(progress_images_path))
    framerate = max(1, number_of_frames // args.progress_video_length)
    process = subprocess.Popen(f"ffmpeg -r {framerate} -f image2 -i {progress_images_path}/%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf pad='width=ceil(iw/2)*2:height=ceil(ih/2)*2' {args.progress_video}", stdout=subprocess.DEVNULL)
    process.wait()
    if args.progress_output is None:
      shutil.rmtree(progress_images_path, ignore_errors=True)
