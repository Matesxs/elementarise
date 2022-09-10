import enum

class MetricsMode(enum.Enum):
  """
  Mode for metrics improvement
  """
  MINIMALISE = 0
  MAXIMALISE = 1

def string_to_metrics_mode(val:str) -> MetricsMode:
  return MetricsMode[val.upper()]

class ElementType(enum.Enum):
  """
  Element selector
  """
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
  """
  Tile select mode
  """
  RANDOM = 0
  ROUND_ROBIN = 1
  PRIORITY = 2
  ONE_BY_ONE = 3
  TARGET = 4

def string_to_tile_select_mode(val:str) -> TileSelectMode:
  return TileSelectMode[val.upper()]
