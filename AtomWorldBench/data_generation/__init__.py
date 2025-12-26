"""
Data generation package for AtomWorldBench.

This package provides tools for generating datasets for AtomWorld tasks, which
typically involve atomic structures and associated actions or properties. The
package includes:

- BaseDataGenerator: An abstract base class that defines the interface for all
  data generators. It provides a consistent API for generating datasets with
  reproducible random sampling. Concrete implementations (like CIFActionGenerator)
  extend this class to generate specific types of data such as CIF-action pairs
  for training and evaluation.

All data generators follow a common pattern: initialize with configuration,
then call generate() to yield data samples as dictionaries.
"""

from .base_data_generator import BaseDataGenerator
