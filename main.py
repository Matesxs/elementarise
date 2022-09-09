import argparse
from PIL import Image
import numpy as np
import os

from elementarise_image import Elementariser

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", "-i", help="Path to input image", type=str, required=True)
  parser.add_argument("--output", "-o", help="Path to output image", type=str, required=True)
  parser.add_argument("--checkpoint", "-ch", help="Path to checkpoint image", type=str, required=False)
  parser.add_argument("--elements", "-e", help="Number of elements to draw (default: 1000)", type=int, default=1000)
  parser.add_argument("--batch_size", "-b", help="Number of elements generated to test (default: 500)", type=int, default=500)
  parser.add_argument("--tries", "-t", help="Limit number of repeats per element (default: 20)", type=int, default=20)
  parser.add_argument("--element_type", "-et", help="Element used for recreating reference image (default: line), line, circle, triangle, square, pentagon, hexagon, octagon, random", type=str, default="random")
  parser.add_argument("--tile_select_mode", "-tsm", help="Tile select mode changes behaviour of tile selection when multiple of them are present (default: random), random - tiles are selected randomly, round_robin - tiles are selected one after another, priority - tiles with worst metrics will get processed first", type=str, default="random")
  parser.add_argument("--process_scale_factor", "-psf", help="Scale down factor for generating image (example: 2 will scale image size in both axis by factor of 2)", type=float, default=1)
  parser.add_argument("--output_scale_factor", "-osf", help="Scale factor for output image (same behaviour as process_scale_factor)", type=float, default=1)
  parser.add_argument("--width_splits", "-ws", help="Number of width splits for generating elements in smaller more specific areas (1 = no splits - default)", type=int, default=1)
  parser.add_argument("--height_splits", "-hs", help="Same as width splits only for height", type=int, default=1)
  parser.add_argument("--workers", "-w", help="Number of workers", type=int, default=2)

  args = parser.parse_args()

  assert os.path.exists(args.input) and os.path.isfile(args.input), "Invalid input file"

  input_image = np.array(Image.open(args.input).convert('RGB'))
  checkpoint_image = None
  if args.checkpoint is not None:
    assert os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint), "Invalid checkpoint file"
    checkpoint_image = np.array(Image.open(args.checkpoint).convert('RGB'))

  elementariser = Elementariser(input_image, checkpoint_image,
                                args.process_scale_factor, args.output_scale_factor,
                                args.elements, args.batch_size, args.tries,
                                args.width_splits, args.height_splits,
                                element_type=args.element_type, tile_select_mode=args.tile_select_mode,
                                workers=args.workers,
                                debug=True, debug_on_progress_image=True, use_tqdm=True, visualise_progress=True)

  final_image = elementariser.run()
  final_image = Image.fromarray(final_image, mode="RGB")
  final_image.save(args.output)
