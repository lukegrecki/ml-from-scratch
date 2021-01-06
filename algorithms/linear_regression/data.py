import random
from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Dict, Generator
import matplotlib.pyplot as plt


@dataclass
class Parameters:
    m: float
    b: float
