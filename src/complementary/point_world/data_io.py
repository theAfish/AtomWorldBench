import h5py
import json
import numpy as np

def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_dataset_to_h5(dataset, filepath):
    """
    dataset: list of dicts, 每个dict包含:
        - 'task' (str)
        - 'state_before' (list of list of floats)
        - 'action_prompt' (str)
        - 'state_after' (list of list of floats)
        - 'params' (dict)
    filepath: 保存的h5文件路径
    """
    with h5py.File(filepath, "w") as f:
        data_group = f.create_group("data")

        for i, sample in enumerate(dataset):
            grp = data_group.create_group(f"sample_{i}")
            grp.attrs["task"] = sample["task"]
            grp.create_dataset("state_before", data=np.array(sample["state_before"], dtype=np.float32))
            grp.create_dataset("state_after", data=np.array(sample["state_after"], dtype=np.float32))

            dt = h5py.string_dtype(encoding='utf-8')
            grp.create_dataset("action_prompt", data=sample["action_prompt"], dtype=dt)

            params_clean = convert_ndarray_to_list(sample["params"])
            params_json = json.dumps(params_clean)
            grp.create_dataset("params", data=params_json, dtype=dt)


def load_dataset_from_h5(filepath, action_name=None):
    dataset = []
    if action_name:
        # the filename should be "filepath/{action_name}_data.h5"
        filepath = filepath if filepath.endswith(".h5") else f"{filepath}/{action_name}_data.h5"
    with h5py.File(filepath, "r") as f:
        data_group = f["data"]
        for sample_name in data_group:
            grp = data_group[sample_name]
            sample = {
                "task": grp.attrs["task"],
                "state_before": grp["state_before"][()],
                "action_prompt": grp["action_prompt"][()].decode('utf-8') if isinstance(grp["action_prompt"][()], bytes) else grp["action_prompt"][()],
                "state_after": grp["state_after"][()],
                "params": json.loads(grp["params"][()].decode('utf-8') if isinstance(grp["params"][()], bytes) else grp["params"][()])
            }
            dataset.append(sample)
    return dataset

