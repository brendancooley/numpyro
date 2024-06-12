from typing import Union

import jax
from jaxlib.xla_extension import ArrayImpl
import numpy as np

ARRAY_TYPE = Union[jax.Array, np.ndarray]  # jax.Array covers tracers
