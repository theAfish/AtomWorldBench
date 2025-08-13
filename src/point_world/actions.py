from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseAction(ABC):
    def __init__(self, action_name: str, **kwargs):
        self.action_name = action_name
        self.config = kwargs

    @abstractmethod
    def execute(self, env: Any) -> Any:
        """
        Execute the action in the given environment.

        Args:
            env (Any): The environment in which to execute the action.
            **kwargs: Additional parameters for the action.

        Returns:
            Any: Result of the action execution.
        """
        pass

    @property
    def name(self) -> str:
        return self.action_name

    def __str__(self):
        return f"Action: {self.action_name}, Config: {self.config}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.action_name}')"


class MoveAction(BaseAction):
    def __init__(self, index: int, displacement: Union[List[float], np.ndarray]):
        super().__init__("move", index=index, displacement=displacement)

    def execute(self, env: Any) -> Any:
        env.move(self.config["index"], np.array(self.config["displacement"]))
        return env.get_state()

    def __str__(self):
        return f"Move the point at index {self.config['index']} by displacement {self.config['displacement']}"


class MoveTowardsAction(BaseAction):
    def __init__(self, from_index: int, to_index: int, distance: float):
        super().__init__("move_towards", from_index=from_index, to_index=to_index, distance=distance)

    def execute(self, env: Any, **kwargs) -> Any:
        env.move_towards(self.config["from_index"], self.config["to_index"], self.config["distance"])
        return env.get_state()

    def __str__(self):
        return f"Move the point at index {self.config['from_index']} towards the point at index {self.config['to_index']} by {self.config['distance']}"


class InsertBetweenAction(BaseAction):
    def __init__(self, index1: int, index2: int, distance: float = 0.5):
        super().__init__("insert_between", index1=index1, index2=index2, distance=distance)

    def execute(self, env: Any) -> Any:
        env.insert_between(self.config["index1"], self.config["index2"], self.config["distance"])
        return env.get_state()

    def __str__(self):
        return (f"Insert a new point between points at indices {self.config['index1']} and {self.config['index2']}, "
                f"{self.config['distance']} units away from point {self.config['index1']}.")


class RotateAroundAction(BaseAction):
    def __init__(self, center_index: int, axis: Union[List[float], np.ndarray], angle_deg: float):
        super().__init__("rotate_around", center_index=center_index, axis=axis, angle_deg=angle_deg)

    def execute(self, env: Any) -> Any:
        env.rotate_around_3D(self.config["center_index"], self.config["axis"], self.config["angle_deg"])
        return env.get_state()

    def __str__(self):
        return (f"Rotate all points by {self.config['angle_deg']} degrees around the axis {self.config['axis']}, "
                f"with the point at index {self.config['center_index']} as the center of rotation. "
                "The rotation follows the right-hand rule.")

