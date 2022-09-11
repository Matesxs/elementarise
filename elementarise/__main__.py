import argparse
from PIL import Image
import numpy as np
import os

from .runtime import Elementariser, AutoElementariser

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", help="Path to input image", type=str, required=True)
parser.add_argument("--output", "-o", help="Path to output image", type=str, required=True)
parser.add_argument("--checkpoint", "-ch", help="Path to checkpoint image", type=str, required=False)
parser.add_argument("--elements", "-e", help="Number of elements to draw (default: 2000)", type=int, default=1000)
parser.add_argument("--batch_size", "-b", help="Number of elements generated to test (default: 200)", type=int, default=200)
parser.add_argument("--tries", "-t", help="Limit number of repeats per element (default: 20)", type=int, default=20)
parser.add_argument("--element_type", "-et", help="Element used for recreating reference image (default: line), line, circle, triangle, square, pentagon, hexagon, octagon, random", type=str, default="random")
parser.add_argument("--min_alpha", help="Minimal alpha value (default: 1)", type=int, default=1)
parser.add_argument("--max_alpha", help="Maximal alpha value (default: 255)", type=int, default=255)
parser.add_argument("--max_size_start_coef", help="Maximum size start coef (default: 0.4)", type=float, default=0.4)
parser.add_argument("--max_size_end_coef", help="Maximum size final coef (default: 0.1)", type=float, default=0.1)
parser.add_argument("--max_size_decay_coef", help="Maximum size decay coef (multiplier for size translation) (default: 1)", type=float, default=1.0)
parser.add_argument("--min_size", help="Minimum size (default: 2)", type=int, default=2)
parser.add_argument("--tile_select_mode", "-tsm", help="Tile select mode changes behaviour of tile selection when multiple of them are present (default: random), random - tiles are selected randomly, round_robin - tiles are selected one after another, priority - tiles with worst metrics will get processed first, one_by_one - tiles will be completed one after another (not good for generating from start), target - target specific tile for processing", type=str, default="random")
parser.add_argument("--target_tile", help="Tile indexes for target tile select mode", required=False, type=str)
parser.add_argument("--process_scale_factor", "-psf", help="Scale down factor for generating image (example: 2 will scale image size in both axis by factor of 2)", type=float, default=1)
parser.add_argument("--output_scale_factor", "-osf", help="Scale factor for output image (same behaviour as process_scale_factor)", type=float, default=1)
parser.add_argument("--width_splits", "-ws", help="Number of width splits for generating elements in smaller more specific areas (1 = no splits - default)", type=int, default=1)
parser.add_argument("--height_splits", "-hs", help="Same as width splits only for height", type=int, default=1)
parser.add_argument("--workers", "-w", help="Number of workers", type=int, default=2)
parser.add_argument("--disable_visuals", action="store_false", help="Disable progress image output")
parser.add_argument("--save_progress", action="store_true", help="Store progress of generation")
parser.add_argument("--progress_folder", help="Path to folder where progress imagis will be saved (default: tmp)", type=str, default="tmp")
parser.add_argument("--automatic", action="store_true", help="Use automatic elementariser (some arguments will not be used)")

args = parser.parse_args()

assert os.path.exists(args.input) and os.path.isfile(args.input), "Invalid input file"

tile_target = None
if args.target_tile is not None:
  target_splits = args.target_tile.split(",")
  tile_target = (int(target_splits[0]), int(target_splits[1]))

input_image = np.array(Image.open(args.input).convert('RGB'))
checkpoint_image = None
if args.checkpoint is not None:
  assert os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint), "Invalid checkpoint file"
  checkpoint_image = np.array(Image.open(args.checkpoint).convert('RGB'))

if args.automatic:
  elementariser = AutoElementariser(input_image, checkpoint_image,
                                    args.output_scale_factor,
                                    args.elements, args.batch_size, args.tries,
                                    element_type=args.element_type,
                                    workers=args.workers,
                                    debug=True, debug_on_progress_image=True, use_tqdm=True, visualise_progress=args.disable_visuals,
                                    progress_save_path=args.progress_folder, save_progress=args.save_progress)
else:
  elementariser = Elementariser(input_image, checkpoint_image,
                                args.process_scale_factor, args.output_scale_factor,
                                args.elements, args.batch_size, args.tries,
                                args.width_splits, args.height_splits,
                                args.min_alpha, args.max_alpha,
                                args.max_size_start_coef, args.max_size_end_coef, args.max_size_decay_coef, args.min_size,
                                element_type=args.element_type, tile_select_mode=args.tile_select_mode, tile_target=tile_target,
                                workers=args.workers,
                                debug=True, debug_on_progress_image=True, use_tqdm=True, visualise_progress=args.disable_visuals,
                                progress_save_path=args.progress_folder, save_progress=args.save_progress)

final_image, _ = elementariser.run()
final_image = Image.fromarray(final_image, mode="RGB")
final_image.save(args.output)
