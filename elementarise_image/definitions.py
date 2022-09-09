import enum

class ElementType(enum.Enum):
  LINE = 0
  CIRCLE = 1
  TRIANGLE = 2
  SQUARE = 3
  PENTAGON = 4
  HEXAGON = 5
  OCTAGON = 6
  RANDOM = 100

def string_to_element_type(val:str) -> ElementType:
  return ElementType[val.upper()]

class TileSelectMode(enum.Enum):
  RANDOM = 0
  ROUND_ROBIN = 1
  PRIORITY = 2

def string_to_tile_select_mode(val:str) -> TileSelectMode:
  return TileSelectMode[val.upper()]
