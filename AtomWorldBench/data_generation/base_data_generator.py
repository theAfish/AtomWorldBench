from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import numpy as np

class BaseDataGenerator(ABC):
    """
    Abstract base class for data generators in AtomWorld.
    
    This class defines the interface for generating datasets for AtomWorld tasks,
    typically involving atomic structures and associated actions or properties.
    """

    def __init__(self, seed: Optional[int] = 75):
        """
        Initialize the data generator.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Generate a sequence of data samples.

        Args:
            num_samples (int): The number of samples to generate.
            **kwargs: Additional arguments for generation.

        Yields:
            Iterator[Dict[str, Any]]: A dictionary representing a single data sample.
                Expected keys might include 'input_structure', 'action', 'output_structure', etc.
        """
        pass
