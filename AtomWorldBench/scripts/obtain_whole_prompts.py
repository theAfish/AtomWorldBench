import os
import pandas as pd
from utils.dataloader import load_data
from prompts.cif_action_prompt import cif_action_prompt


def get_prompts(action_name, data_folder, num_prompts=20, save_folder: str = "prompts"):
    os.makedirs(save_folder, exist_ok=True)
    data = load_data(data_folder, action_name)

    prompts = []
    outs = []
    results = []
    current_prompt_count = 0
    for i, row in data.iterrows():
        input_cif = row['input_cif']
        action_prompt = row['action_prompt']
        output_cif = row['output_cif']

        # check if the input_cif is empty
        if input_cif is None:
            raise ValueError(f"input_cif is None at row {i}")

        prompt = cif_action_prompt(
            input_cif=input_cif,
            action_prompt=action_prompt,
            output_format="cif"
        )
        prompts.append(prompt)
        outs.append(output_cif)

        results.append({
            "input_cif": input_cif,
            "prompt": prompt,
            "output_cif": output_cif,
            
        })

        current_prompt_count += 1
        if current_prompt_count >= num_prompts:
            break


    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(save_folder, f"{action_name}_prompt_out_{num_prompts}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Prompts saved to {results_csv_path}")


if __name__ == "__main__":

    get_prompts("insert_between_atoms_action", "D:\AI\PythonProjects\AtomWorldBench\src\data", num_prompts=20, save_folder="prompts")