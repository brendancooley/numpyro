from typing import Union

from jaxlib.xla_extension import ArrayImpl
import numpy as np

ARRAY_TYPE = Union[ArrayImpl, np.ndarray]
