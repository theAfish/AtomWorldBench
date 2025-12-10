import datetime
import io
import os
import json
from pathlib import Path
from utils.load_model import load_model, load_config

CONFIG_DIR = Path(__file__).parent.parent / "config"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results/json"

prompt_input = (
    "[INST] <<SYS>>\n"
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    "<</SYS>> \n\n {instruction} \n{input} [/INST]"
)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def run_inference(
        model_id: str, 
        test_path: str,
        batch_size: int,
        config_name: str="models", 
        results_folder: str=None
    ):
    
    config = load_config(CONFIG_DIR / config_name)[model_id]
    
    # Initialize model
    model = load_model(config)

    # automatically set results folder if not provided
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"{results_folder or RESULTS_DIR}/{model_id}/{timestamp}"
    
    list_data_dict = jload(test_path)
    prompts = [
        prompt_input.format_map(example) for example in list_data_dict
    ]

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : min(i + batch_size, len(prompts))]

        batch_original_data = list_data_dict[i : min(i + batch_size, len(list_data_dict))]
        output_batch = model.generate_batch(batch_prompts)
        for original_example, generated_text in zip(batch_original_data, output_batch):
            # Create a copy to ensure we don't accidentally mutate the original dataset in memory
            result_entry = original_example.copy()
            
            # Overwrite or add the 'output' key with the model's response
            result_entry['output'] = generated_text
            
            results.append(result_entry)
    
    output_path = results_folder + "/results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark with specified configuration.")
    parser.add_argument(
        "-m",
        "--model", 
        type=str, 
        default="deepseek_reasoner", 
        help="ID of the model to use (e.g., 'deepseek_chat', 'openai_gpt4')"
    )
    parser.add_argument(
        "--test_path", 
        type=str, 
        help=f"Path to test dataset (json)"
    )
    parser.add_argument(
        "-c",
        "--config", 
        type=str, 
        default="models", 
        help=f"Name of config file (located under src/config)"
    )
    parser.add_argument(
        "-b",
        "--batch_size", 
        type=int, 
        default=50, 
        help=f"Batch size for the run. Default: 50"
    )
    
    args = parser.parse_args()
    
    run_inference(
        args.model, 
        args.test_path,
        batch_size=args.batch_size, 
        config_name=args.config
    )
