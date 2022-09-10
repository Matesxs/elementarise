import numpy as np
from PIL import Image, ImageDraw
import typing

from .helpers import image_to_numpy, numpy_to_image, get_line_boundingbox, calculate_polygon_coords
from .definitions import ElementType

def draw_polygon(image: np.ndarray, params:tuple, polygon_count:int) -> tuple:
  height, width, _ = image.shape

  color = params[4]
  overlay = Image.new("RGBA", (width, height), color[:3] + (0,))
  draw = ImageDraw.Draw(overlay)

  coords = calculate_polygon_coords(params[0], params[1], params[2], params[3], polygon_count)

  x_coords = []
  y_coords = []
  for idx in range(polygon_count * 2):
    if idx % 2 == 0:
      x_coords.append(coords[idx])
    else:
      y_coords.append(coords[idx])

  draw.polygon([(xc, yc) for xc, yc in zip(x_coords, y_coords)], fill=color)
  bbox = np.array([max(0, min(x_coords) - 1), max(0, min(y_coords) - 1), min(width, max(x_coords) + 1), min(height, max(y_coords) + 1)], dtype=int)
  return overlay, bbox

def draw_element(output_image:np.ndarray, params:tuple, element_type:ElementType) -> typing.Tuple[np.ndarray, np.ndarray]:
  height, width, _ = output_image.shape

  img = numpy_to_image(output_image)

  if element_type == ElementType.LINE:
    overlay = Image.new("RGBA", img.size, params[5][:3] + (0,))
    draw = ImageDraw.Draw(overlay)
    end_point = (params[0] + np.cos(params[4]) * params[2], params[1] + np.sin(params[4]) * params[2])
    coords = [params[0], params[1], *end_point]
    draw.line(coords, params[5], params[3])
    bbox = get_line_boundingbox(np.array(coords, dtype=np.float64), params[3], width, height).astype(int)
  elif element_type == ElementType.CIRCLE:
    thickness = (params[2] - 1) / 2
    color = params[3]
    ellipse_params = (params[0] - thickness, params[1] - thickness, params[0] + thickness, params[1] + thickness)

    overlay = Image.new("RGBA", img.size, color[:3] + (0,))
    draw = ImageDraw.Draw(overlay)

    draw.ellipse(ellipse_params, fill=color)
    bbox = np.array([max(0, ellipse_params[0] - 1), max(0, ellipse_params[1] - 1), min(width, ellipse_params[2] + 1), min(height, ellipse_params[3] + 1)], dtype=int)
  elif element_type == ElementType.TRIANGLE:
    overlay, bbox = draw_polygon(output_image, params, 3)
  elif element_type == ElementType.SQUARE:
    overlay, bbox = draw_polygon(output_image, params, 4)
  elif element_type == ElementType.PENTAGON:
    overlay, bbox = draw_polygon(output_image, params, 5)
  elif element_type == ElementType.HEXAGON:
    overlay, bbox = draw_polygon(output_image, params, 6)
  elif element_type == ElementType.OCTAGON:
    overlay, bbox = draw_polygon(output_image, params, 8)
  else:
    raise Exception("Invalid element type for draw")

  img.alpha_composite(overlay)

  return image_to_numpy(img), bbox
