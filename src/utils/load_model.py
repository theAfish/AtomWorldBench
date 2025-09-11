import os
from models.openai_model import OpenAIModel
from models.azure_openai_model import AzureOpenAIModel
from models.huggingface_model import HuggingFaceModel
from models.vllm_model import vllmModel


def load_model(config):
    model_class = config.get("class")
    if model_class == "OpenAIModel":
        api_key = os.path.expandvars(config.get("api_key", ""))
        model = OpenAIModel(
            model_name=config['model_name'],
            api_key=api_key,
            base_url=config.get('base_url'),
            temperature=config.get('temperature', 1)
        )
    elif model_class == "AzureOpenAIModel":
        model_name = os.path.expandvars(config.get("model_name", ""))
        api_key = os.path.expandvars(config.get("api_key", ""))
        api_version = os.path.expandvars(config.get("api_version", ""))
        azure_endpoint = os.path.expandvars(config.get("azure_endpoint", ""))

        model = AzureOpenAIModel(
            model_name=model_name,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            temperature=config.get('temperature', 1)
        )
    elif model_class == "HuggingFaceModel":
        model_name = config.get("model_name", None)
        device = config.get("device", "cpu")
        use_pipeline = config.get("use_pipeline", True)
        generation_params = {k: v for k, v in config.items() if
                             k not in ["class", "model_name", "device", "use_pipeline"]}

        model = HuggingFaceModel(
            model_name=model_name,
            device=device,
            use_pipeline=use_pipeline,
            **generation_params
        )
    elif model_class == "vllmModel":
        model_name = config.get("model_name", None)
        generation_params = {k: v for k, v in config.items() if k not in ["class", "model_name", "device", "use_pipeline"]}
        
        model = vllmModel(
            model_name=model_name,
            **generation_params
        )
    else:
        raise ValueError(f"Unimplemented model_class '{model_class}'.")
    
    return model