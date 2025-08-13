from point_world.data_generator import PointWorldDataGenerator
from point_world.data_io import save_dataset_to_h5, load_dataset_from_h5


generator = PointWorldDataGenerator(dim=3, num_points=2, gen_limit=10.0, seed=42)
tasks = ["insert_between"]
for task in tasks:
    dataset = generator.generate(task, N=1000)
    print(f"Generated {len(dataset)} samples for task '{task}'")
    save_dataset_to_h5(dataset, f"{task}_data.h5")
    # for sample in data:
    #     print(sample)
