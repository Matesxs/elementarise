# Elementarise image

## Description
Library for generating images from basic shapes.

### Example
![o1](https://github.com/Matesxs/elementarise/blob/master/media/cat.jpg?raw=true "Original")
![g1](https://github.com/Matesxs/elementarise/blob/master/media/cat_result.png?raw=true "Generated")  
![o2](https://github.com/Matesxs/elementarise/blob/master/media/eevee.png?raw=true "Original")
![g2](https://github.com/Matesxs/elementarise/blob/master/media/eevee_result.png?raw=true "Generated")  
![o3](https://github.com/Matesxs/elementarise/blob/master/media/portal.png?raw=true "Original")
![g3](https://github.com/Matesxs/elementarise/blob/master/media/portal_result.png?raw=true "Generated")


## How it works
The library is guessing parameters of selected element and placing it on process image with intension of matching the reference image as good as possible.


## Setup
### Instalation
```
pip install elementarise
```

### Development
```
python setup.py develop
```


## Usage
### Programatic use
#### Parameter list
```python
Elementariser(
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
  visualise_progress:bool=False
)
```

#### Example use

```python
from elementarise import Elementariser, ElementType
from PIL import Image
import numpy as np

reference_image = np.array(Image.open("reference_image.png", mode="RGB"))

elementariser = Elementariser(
  reference_image,
  process_scale_factor=0.5,
  output_scale_factor=2.0,
  num_of_elements=5000,
  batch_size=200,
  num_of_retries=50,
  width_divs=2,
  height_divs=2,
  element_type=ElementType.CIRCLE,
  workers=4,
  use_tqdm=True
)

output_image = elementariser.run()
Image.fromarray(reference_image, mode="RGB").save("result.png")
```

### Script
```
python -m elementarise --help

usage: __main__.py [-h] --input INPUT --output OUTPUT [--checkpoint CHECKPOINT] [--elements ELEMENTS] [--batch_size BATCH_SIZE] [--tries TRIES]
                   [--element_type ELEMENT_TYPE] [--min_alpha MIN_ALPHA] [--max_alpha MAX_ALPHA] [--max_size_start_coef MAX_SIZE_START_COEF]
                   [--max_size_end_coef MAX_SIZE_END_COEF] [--max_size_decay_coef MAX_SIZE_DECAY_COEF] [--min_size MIN_SIZE]
                   [--tile_select_mode TILE_SELECT_MODE] [--target_tile TARGET_TILE] [--process_scale_factor PROCESS_SCALE_FACTOR]
                   [--output_scale_factor OUTPUT_SCALE_FACTOR] [--width_splits WIDTH_SPLITS] [--height_splits HEIGHT_SPLITS] [--workers WORKERS]
                   [--disable_visuals] [--save_progress] [--progress_folder PROGRESS_FOLDER]

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to input image
  --output OUTPUT, -o OUTPUT
                        Path to output image
  --checkpoint CHECKPOINT, -ch CHECKPOINT
                        Path to checkpoint image
  --elements ELEMENTS, -e ELEMENTS
                        Number of elements to draw (default: 2000)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Number of elements generated to test (default: 200)
  --tries TRIES, -t TRIES
                        Limit number of repeats per element (default: 20)
  --element_type ELEMENT_TYPE, -et ELEMENT_TYPE
                        Element used for recreating reference image (default: line), line, circle, triangle, square, pentagon, hexagon, octagon, random
  --min_alpha MIN_ALPHA
                        Minimal alpha value (default: 1)
  --max_alpha MAX_ALPHA
                        Maximal alpha value (default: 255)
  --max_size_start_coef MAX_SIZE_START_COEF
                        Maximum size start coef (default: 0.4)
  --max_size_end_coef MAX_SIZE_END_COEF
                        Maximum size final coef (default: 0.1)
  --max_size_decay_coef MAX_SIZE_DECAY_COEF
                        Maximum size decay coef (multiplier for size translation) (default: 1)
  --min_size MIN_SIZE   Minimum size (default: 2)
  --tile_select_mode TILE_SELECT_MODE, -tsm TILE_SELECT_MODE
                        Tile select mode changes behaviour of tile selection when multiple of them are present (default: random), random - tiles are
                        selected randomly, round_robin - tiles are selected one after another, priority - tiles with worst metrics will get processed first,
                        one_by_one - tiles will be completed one after another (not good for generating from start), target - target specific tile for
                        processing
  --target_tile TARGET_TILE
                        Tile indexes for target tile select mode
  --process_scale_factor PROCESS_SCALE_FACTOR, -psf PROCESS_SCALE_FACTOR
                        Scale down factor for generating image (example: 2 will scale image size in both axis by factor of 2)
  --output_scale_factor OUTPUT_SCALE_FACTOR, -osf OUTPUT_SCALE_FACTOR
                        Scale factor for output image (same behaviour as process_scale_factor)
  --width_splits WIDTH_SPLITS, -ws WIDTH_SPLITS
                        Number of width splits for generating elements in smaller more specific areas (1 = no splits - default)
  --height_splits HEIGHT_SPLITS, -hs HEIGHT_SPLITS
                        Same as width splits only for height
  --workers WORKERS, -w WORKERS
                        Number of workers
  --disable_visuals     Disable progress image output
  --save_progress       Store progress of generation
  --progress_folder PROGRESS_FOLDER
                        Path to folder where progress imagis will be saved (default: tmp)
```
