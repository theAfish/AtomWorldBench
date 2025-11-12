#!/usr/bin/env python3
"""Test script for DataGeneratorV2."""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from data_generator_v2 import DataGeneratorV2
from actions_v2 import (
    AddAtomActionV2,
    RemoveAtomActionV2,
    MoveAtomActionV2,
    ChangeAtomActionV2,
    SwapAtomsActionV2,
    InsertBetweenAtomsActionV2,
    MoveTowardsAtomActionV2,
    DeleteBelowAtomActionV2,
    DeleteAroundAtomActionV2,
    MoveSelectedAtomsActionV2,
    MoveAroundAtomActionV2,
    RotateAroundAtomActionV2,
)

def main():
    # Set paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_dir = os.path.join(project_root, "debug", "cifs")
    output_dir = os.path.join(project_root, "debug")

    # Create generator
    generator = DataGeneratorV2(input_dir=input_dir, output_dir=output_dir)

    # Select a few actions to test
    action_classes = [
        # AddAtomActionV2,
        # RemoveAtomActionV2,
        # MoveAtomActionV2,
        # ChangeAtomActionV2,
        # SwapAtomsActionV2,
        # InsertBetweenAtomsActionV2,
        # MoveTowardsAtomActionV2,
        # DeleteBelowAtomActionV2,
        # DeleteAroundAtomActionV2,
        MoveSelectedAtomsActionV2,
        # MoveAroundAtomActionV2,
        # RotateAroundAtomActionV2,
    ]

    # Generate data
    print("Starting data generation...")
    generator.generate_data(action_classes)
    print("Data generation completed.")

if __name__ == "__main__":
    main()
