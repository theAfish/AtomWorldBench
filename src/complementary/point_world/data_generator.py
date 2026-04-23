from complementary.point_world.env import PointWorldEnv
from complementary.point_world.actions import *
import numpy as np
import random

fixed_axes = [
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0]),
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
]


class PointWorldDataGenerator:
    def __init__(self, dim=3, num_points=2, gen_limit=10.0, seed=None, decimals=2):
        self.dim = dim
        self.num_points = num_points
        self.gen_limit = gen_limit
        self.decimals = decimals
        self.rng = np.random.default_rng(seed)

    def _random_env(self):
        env = PointWorldEnv(self.dim, self.num_points, self.gen_limit, self.decimals)
        return env

    def _random_vector(self, scale=1.0):
        vec = self.rng.normal(size=self.dim)
        vec = vec / np.linalg.norm(vec) * scale
        return np.round(vec, decimals=self.decimals)

    def generate(self, task_name, N=10):
        dataset = []
        for _ in range(N):
            env = self._random_env()
            before_state = env.get_state()

            if task_name == "move":
                idx = self.rng.integers(0, env.num_points)
                displacement = self._random_vector(scale=self.rng.uniform(0.5, 5.0))
                params = {"index": int(idx), "displacement": displacement}
                action = MoveAction(**params)
                action.execute(env)

            elif task_name == "move_towards":
                from_idx, to_idx = self.rng.choice(env.num_points, size=2, replace=False)
                distance = np.round(self.rng.uniform(0.5, 5.0), self.decimals)
                params = {"from_index": int(from_idx), "to_index": int(to_idx), "distance": distance}
                action = MoveTowardsAction(**params)
                action.execute(env)

            elif task_name == "insert_between":
                idx1, idx2 = self.rng.choice(env.num_points, size=2, replace=False)
                direction = env.points[idx2] - env.points[idx1]
                norm = np.linalg.norm(direction)
                min_dist = 0.1
                max_dist = norm - 0.1
                distance = np.round(self.rng.uniform(min_dist, max_dist), self.decimals)
                params = {"index1": int(idx1), "index2": int(idx2), "distance": distance}
                action = InsertBetweenAction(**params)
                action.execute(env)

            elif task_name == "rotate_around":
                center_idx = self.rng.integers(0, env.num_points)
                axis = self.rng.choice(fixed_axes)
                angle_deg = np.round(self.rng.uniform(15, 180), self.decimals)
                env.rotate_around_3D(center_idx, np.array(axis), angle_deg)
                params = {"center_index": int(center_idx), "axis": axis, "angle_deg": angle_deg}
                action = RotateAroundAction(**params)
                action.execute(env)

            else:
                raise ValueError(f"Unknown task: {task_name}")

            after_state = env.get_state()
            action_prompt = str(action)

            dataset.append({
                "task": task_name,
                "state_before": before_state.tolist(),
                "action_prompt": action_prompt,
                "state_after": after_state.tolist(),
                "params": params
            })

        return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    def plot_points(state_before, state_after):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 转换为numpy数组，方便计算比例
        before = np.array(state_before)
        after = np.array(state_after)

        # 画初始点（蓝色）
        ax.scatter(before[:, 0], before[:, 1], before[:, 2], c='blue', label='Before')
        for i, pt in enumerate(before):
            ax.text(pt[0], pt[1], pt[2], str(i), color='blue')

        # 画末态点（红色）
        ax.scatter(after[:, 0], after[:, 1], after[:, 2], c='red', label='After')
        for i, pt in enumerate(after):
            ax.text(pt[0], pt[1], pt[2], str(i), color='red')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # ax.set_title(title)
        ax.legend()

        # 设定坐标轴等比例显示
        all_points = np.vstack([before, after])
        max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()

    generator = PointWorldDataGenerator(dim=3, num_points=2, gen_limit=10.0, seed=42)
    # tasks = ["move", "move_towards", "insert_between", "rotate_around"]
    tasks = ["insert_between"]
    for task in tasks:
        data = generator.generate(task, N=20)
        print(f"Generated {len(data)} samples for task '{task}':")
        for sample in data:
            print(sample)
            # plot_points(sample["state_before"], sample["state_after"])
        print()