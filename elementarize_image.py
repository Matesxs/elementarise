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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import shutil
import subprocess
import pathlib

MIN_DIFF = 5_000

@njit(nogil=True)
def get_random_color(min_alpha, max_alpha):
  return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(min_alpha, max_alpha)

@njit(nogil=True)
def get_circle_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha):
  return random.randint(min_width, max_width - 1), \
         random.randint(min_height, max_height - 1), \
         random.randint(1, max_size), \
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

@njit(nogil=True)
def get_line_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha):
  x_coors = np.random.randint(min_width, max_width, size=2)
  y_coors = np.random.randint(min_height, max_height, size=2)
  return x_coors[0], \
         y_coors[0], \
         x_coors[1], \
         y_coors[1], \
         random.randint(1, max_size), \
         get_random_color(min_alpha, max_alpha)

def calculate_coords(x_center, y_center, radius, starting_angle, number_of_vertices):
  angle_increment = np.pi * 2 / number_of_vertices
  return np.array([(int(x_center + np.cos(starting_angle + i * angle_increment) * radius),
                    int(y_center + np.sin(starting_angle + i * angle_increment) * radius)) for i in range(number_of_vertices)], dtype=int).flatten()

@njit(nogil=True)
def get_symmetrical_polygon_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha):
  radius = random.randint(1, max_size)
  x_center = random.randint(min_width, max_width - 1)
  y_center = random.randint(min_height, max_height - 1)
  starting_angle = random.random() * np.pi * 2
  return x_center, \
         y_center, \
         radius, \
         starting_angle, \
         get_random_color(min_alpha, max_alpha)

def get_random_element_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha, mode):
  if mode == 0:
    return get_line_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha)
  elif mode == 1:
    return get_circle_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha)
  elif mode == 2:
    return get_ellipse_params(min_width, max_width, min_height, max_height, min_alpha, max_alpha)
  else:
    return get_symmetrical_polygon_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha)

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
    draw.line(params[:4], params[5], params[4])
    bbox = get_line_boundingbox(np.array(params[:4], dtype=np.float64), params[4], width, height).astype(int)
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
def get_sum_distance(original, fake):
  return np.sum((original - fake) ** 2)

@njit(nogil=True)
def get_mean_distance(original, fake):
  return np.mean((original - fake) ** 2)

@njit(nogil=True)
def get_window_distances(original_window, output_window, current_window):
  prev_distance = get_sum_distance(original_window, output_window)
  new_distance = get_sum_distance(original_window, current_window)
  return prev_distance, new_distance

def get_params(max_size, original_image, output_image, mode, min_width, max_width, min_height, max_height, min_alpha, max_alpha, _):
  params = get_random_element_params(min_width, max_width, min_height, max_height, max_size, min_alpha, max_alpha, mode)
  complete_params = [params, output_image, mode]
  tmp_image, bounding_box = draw_element(*complete_params)
  prev_distance, new_distance = get_window_distances(original_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                                     output_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :],
                                                     tmp_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :])

  # data = np.array(tmp_image)
  # cv2.rectangle(data, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color=(255, 0, 0))
  # cv2.imshow("test", data)
  # cv2.waitKey(0)
  return prev_distance - new_distance, params

@njit(nogil=True)
def translate(value, leftMin, leftMax, rightMin, rightMax):
  leftSpan = leftMax - leftMin
  rightSpan = rightMax - rightMin
  valueScaled = float(value - leftMin) / float(leftSpan)
  return rightMin + (valueScaled * rightSpan)

def generate_output_image(output_image, params_history, scale_factor, save_progress_path):
  progress_images = []

  index_offset = 0
  if save_progress_path is not None:
    if not os.path.exists(save_progress_path):
      os.mkdir(save_progress_path)
    else:
      files = [pathlib.Path(p).stem for p in os.listdir(save_progress_path)]
      indexes = [int(f) for f in files if f.isnumeric()]
      index_offset = max(indexes) + 1

  print("Generating final image")
  for idx, param in tqdm(enumerate(params_history), total=len(params_history)):
    mode = param[0]
    element_params = param[1]

    if mode == 0:
      params_to_scale = 5
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
      image = Image.fromarray(output_image, mode="RGB")
      if save_progress_path is not None:
        image.save(f"{save_progress_path}/{idx + index_offset}.png")

  return Image.fromarray(output_image, mode="RGB"), progress_images

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", "-i", help="Path to input image", type=str, required=True)
  parser.add_argument("--elements", "-e", help="Number of elements to draw", type=int, default=1000)
  parser.add_argument("--tries", "-t", help="Number of tries for each element", type=int, default=500)
  parser.add_argument("--repeats_limit", "-rl", help="Limit number of repeats per element", type=int, default=20)
  parser.add_argument("--size_multiplier", "-sm", help="Multiplier of size in connection to image (split) dimensions (size dont apply to mode 2)", type=float, default=0.4)
  parser.add_argument("--size_decay_min", "-sdm", help="Minimum element size to which will size decay overtime (if not set no decay will happen) (size dont apply to mode 2)", type=int, required=False)
  parser.add_argument("--size_decay_coef", "-sdc", help="Coefficient of size decay", type=float, default=1)
  parser.add_argument("--min_alpha", "-mina", help="Minimal alpha value of element (default 1) (can't be less than 1)", type=int, default=1)
  parser.add_argument("--max_alpha", "-maxa", help="Maximum alpha value of element (default 255) (can't be more than 255)", type=int, default=255)
  parser.add_argument("--mode", "-m", help="Select element which will be generated (0 - line/rectangle, 1 - circle, 2 - ellipse, 3 - triangles, 4 - squares, 5 - pentagon, bigger mode values will generate coresponding polygon)", type=int, default=0)
  parser.add_argument("--width_splits", "-ws", help="Number of width splits for generating elements in smaller more specific areas (1 = no splits - default)", type=int, default=1)
  parser.add_argument("--height_splits", "-hs", help="Same as width splits only for height", type=int, default=1)
  parser.add_argument("--target_high_distance_splits", "-thds", action="store_true", help="Target zones with highest distance from input images (works only when using splits")
  parser.add_argument("--workers", "-w", help="Number of workers to serach for solution", type=int, default=4)
  parser.add_argument("--progress", "-p", action="store_true", help="Show progress")
  parser.add_argument("--output", "-o", help="Path where to save output image", type=str, required=False)
  parser.add_argument("--progress_output", "-po", help="Path to folder where progress images will be saved", type=str, required=False)
  parser.add_argument("--progress_video", "-pv", help="Path to video of progress (works only with ffmpeg installed and in PATH) (if progress_output is not defined temporary folder with images will be created)", type=str, required=False)
  parser.add_argument("--progress_video_framerate", "-pvf", help="Framerate of output video (more frames per second -> faster video) (default: 100)", type=int, default=100)
  parser.add_argument("--process_scale_factor", "-psf", help="Scale down factor for generating image (example: 2 will scale image size in both axis by factor of 2)", type=float, default=1)
  parser.add_argument("--output_scale_factor", "-osf", help="Scale factor for output image (same behaviour as process_scale_factor)", type=float, default=1)
  parser.add_argument("--checkpoint", "-ch", help="Checkpoint image path", type=str, required=False)
  parser.add_argument("--random_mode", "-rm", action="store_true", help="Mode specifier gets ignored and mode is selected randomly for each element generation (modes 0 - 5 are used)")

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
  assert args.progress_video_framerate >= 1, "Invalid progress video framerate"
  assert args.size_decay_coef > 0, "Invalid size decay coefficient"

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

  process_image_data = np.zeros_like(resized_input_image)
  if args.checkpoint is not None:
    checkpoint_image = Image.open(args.checkpoint).convert("RGB")
    checkpoint_image = np.array(checkpoint_image)
    process_image_data = cv2.resize(checkpoint_image, dsize=(process_image_data.shape[1], process_image_data.shape[0]), interpolation=cv2.INTER_AREA)
    output_image = checkpoint_image.copy() # We don't resize because we expect that this is wanted size and resizing will only introduce artifacts and image distortion

  width_split_coef = math.ceil(resized_width / args.width_splits)
  height_split_coef = math.ceil(resized_height / args.height_splits)
  max_size = min(width_split_coef, height_split_coef) * args.size_multiplier
  default_max_thickness = max_size = max(1, max_size)
  print(f"Tile size: {width_split_coef}x{height_split_coef}")

  if args.size_decay_min is not None:
    assert 1 <= args.size_decay_min <= max_size, f"Invalid decay minimum size, maximum is {max_size} for current settings"

  line_params_history = []
  executor = ThreadPoolExecutor(args.workers)
  start_time = time.time()
  line_index = 0
  _indexes = list(range(args.tries))

  try:
    iterator = tqdm(range(args.elements))
    for line_index in iterator:
      repeats = 0

      mode = args.mode if not args.random_mode else random.randint(0, 5)
      min_width, max_width = 0, resized_width
      min_height, max_height = 0, resized_height
      selected_width_reg = 0
      selected_height_reg = 0
      if args.width_splits > 1 or args.height_splits > 1:
        if args.target_high_distance_splits:
          distances = [get_mean_distance(
            resized_input_image[height_split_coef * iy:min(resized_height, height_split_coef * (iy + 1)) - 1, width_split_coef * ix:min(resized_width, width_split_coef * (ix + 1)) - 1, :],
            process_image_data[height_split_coef * iy:min(resized_height, height_split_coef * (iy + 1)) - 1, width_split_coef * ix:min(resized_width, width_split_coef * (ix + 1)) - 1, :]
          ) for iy in range(args.height_splits) for ix in range(args.width_splits)]
          distances = np.array(distances, dtype=int).reshape((args.height_splits, args.width_splits))
          selected_height_reg, selected_width_reg = np.unravel_index(distances.argmax(), distances.shape)
        else:
          selected_width_reg = random.randint(0, args.width_splits - 1)
          selected_height_reg = random.randint(0, args.height_splits - 1)

        min_width = width_split_coef * selected_width_reg
        max_width = min(resized_width, width_split_coef * (selected_width_reg + 1))

        min_height = height_split_coef * selected_height_reg
        max_height = min(resized_height, height_split_coef * (selected_height_reg + 1))

      get_params_function = partial(get_params, int(max_size), resized_input_image, process_image_data, mode, min_width, max_width, min_height, max_height, args.min_alpha, args.max_alpha)

      while True:
        scored_params = list(executor.map(get_params_function, _indexes))
        scored_params.sort(key=lambda x: x[0], reverse=True)

        distance_diff = scored_params[0][0]

        if distance_diff > MIN_DIFF:
          break
        else:
          repeats += 1
          if repeats >= args.repeats_limit:
            break
      if repeats >= args.repeats_limit:
        print("Retries limit reached, ending")
        break

      params = scored_params[0][1]
      line_params_history.append((mode, params))
      params = [params, process_image_data, mode]
      process_image_data, bbox = draw_element(*params)
      if args.size_decay_min is not None:
        max_size = max(args.size_decay_min, translate(args.size_decay_coef * line_index, 0, args.elements, default_max_thickness, args.size_decay_min))

      if args.progress:
        # print(f"Distance: {get_distance(original_image, output_image_data)}, Improvement: {distance_diff}")
        iterator.set_description(f"Distance: {get_sum_distance(resized_input_image, process_image_data)}, Improvement: {distance_diff}, Max size: {int(max_size)}")

        prog_image = cv2.cvtColor(process_image_data, cv2.COLOR_RGB2BGR)
        for xidx in range(args.width_splits):
          x1 = width_split_coef * xidx
          x2 = min(resized_width, width_split_coef * (xidx + 1))
          for yidx in range(args.height_splits):
            y1 = height_split_coef * yidx
            y2 = min(resized_height, height_split_coef * (yidx + 1))
            cv2.rectangle(prog_image, (x1, y1), (x2 - 1, y2 - 1), color=(255, 0, 0))

        cv2.rectangle(prog_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 150, 0))
        cv2.rectangle(prog_image, (min_width, min_height), (max_width - 1, max_height - 1), color=(0, 200, 255))

        cv2.imshow("progress_window", prog_image)
        cv2.setWindowTitle("progress_window", f"Progress {line_index + 1}/{args.elements}")
        cv2.waitKey(1)
  except KeyboardInterrupt:
    print("Interrupted by user")

  cv2.destroyAllWindows()
  executor.shutdown()
  print(f"{line_index + 1} elements, Elapsed: {(time.time() - start_time) / 60}mins")

  progress_images_path = args.progress_output if args.progress_output is not None else ("tmp" if args.progress_video is not None else None)
  output_image_object, progress_images = generate_output_image(output_image, line_params_history, output_image.shape[1] / resized_width, progress_images_path)

  if args.output is not None:
    output_image_object.save(args.output)

  if args.progress_video is not None:
    print("Generating progress video")
    process = subprocess.Popen(f"ffmpeg -r {args.progress_video_framerate} -f image2 -i {progress_images_path}/%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {args.progress_video}", stdout=subprocess.DEVNULL)
    process.wait()
    if args.progress_output is None:
      shutil.rmtree(progress_images_path, ignore_errors=True)
