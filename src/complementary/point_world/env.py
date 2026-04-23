from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Union



class PointWorldEnv:
    def __init__(
            self,
            dim: int = 3,
            num_points: int = 2,
            gen_limit: float = 10.0,
            min_distance: float = 0.5,
            decimals: int = 2
    ):
        self.dim = dim
        self.num_points = num_points
        self.gen_limit = gen_limit
        self.min_distance = min_distance
        self.decimals = decimals
        self.points = self._initialize_points()

    def _initialize_points(self):
        points = []
        while len(points) < self.num_points:
            candidate = np.random.uniform(-self.gen_limit, self.gen_limit, self.dim)
            if all(np.linalg.norm(candidate - p) >= self.min_distance for p in points):
                points.append(candidate)
        return np.round(points, decimals=self.decimals)

    def reset(self):
        self.points = self._initialize_points()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.points.copy()

    def move(self, index, displacement: np.ndarray):
        if index < 0 or index >= self.num_points:
            raise IndexError("Point index out of range.")
        if displacement.shape != (self.dim,):
            raise ValueError(f"Displacement must be of shape ({self.dim},).")
        self.points[index] += displacement

    def move_towards(self, from_index, to_index, displacement: float):
        if from_index < 0 or from_index >= self.num_points:
            raise IndexError("from_index out of range.")
        if to_index < 0 or to_index >= self.num_points:
            raise IndexError("to_index out of range.")
        direction = self.points[to_index] - self.points[from_index]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Points are at the same position; cannot determine direction.")
        direction /= norm
        new_pos = self.points[from_index] + direction * displacement
        self.points[from_index] = np.round(new_pos, decimals=self.decimals)

    def insert_between(self, index1, index2, distance: float = 0.5):
        if index1 < 0 or index1 >= self.num_points:
            raise IndexError("index1 out of range.")
        if index2 < 0 or index2 >= self.num_points:
            raise IndexError("index2 out of range.")
        direction = self.points[index2] - self.points[index1]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Points are at the same position; cannot determine direction.")
        direction /= norm
        new_point = np.round(direction * distance + self.points[index1], self.decimals)
        self.points = np.vstack([self.points, new_point])
        self.num_points += 1

    # for 3D only
    def rotate_around_3D(self, center_index, axis: np.ndarray, angle_deg: float):
        if center_index < 0 or center_index >= self.num_points:
            raise IndexError("center_index out of range.")
        if axis.shape != (self.dim,):
            raise ValueError(f"Axis must be of shape ({self.dim},).")

        angle = np.deg2rad(angle_deg)

        axis = axis / np.linalg.norm(axis)
        center_point = self.points[center_index]

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        relative_positions = self.points - center_point
        rotated_positions = relative_positions @ rotation_matrix.T
        self.points = np.round(center_point + rotated_positions, self.decimals)




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    env = PointWorldEnv(dim=3, num_points=20, gen_limit=5.0)
    original_points = env.get_state().copy()

    center_idx = 0
    axis = np.array([0, 1, 0])
    angle = 45
    env.rotate_around_3D(center_idx, axis, angle)
    rotated_points = env.get_state()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
               color='blue', label='Before')

    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
               color='red', label='After')

    center_point = original_points[center_idx]
    ax.scatter(center_point[0], center_point[1], center_point[2],
               color='black', s=100, label='Center')

    for p_before, p_after in zip(original_points, rotated_points):
        ax.plot([p_before[0], p_after[0]],
                [p_before[1], p_after[1]],
                [p_before[2], p_after[2]], color='gray', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
