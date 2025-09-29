import os
import tempfile
import yaml
import pytest

from utils.load_model import load_config, load_model


def test_load_config_tmp_file(tmp_path, monkeypatch):
    data = {"class": "HuggingFaceModel", "model_name": "hf-model"}
    cfg_file = tmp_path / "mycfg.yaml"
    cfg_file.write_text(yaml.safe_dump(data))

    # monkeypatch cwd to tmp_path so load_config finds the file by name
    monkeypatch.chdir(tmp_path)
    loaded = load_config("mycfg")
    assert loaded["class"] == "HuggingFaceModel"
    assert loaded["model_name"] == "hf-model"


def test_load_model_unimplemented_class():
    with pytest.raises(ValueError):
        load_model({"class": "NonExistentModel"})


# We avoid instantiating real model classes (they may require network/large deps).
# Instead, test that load_model raises or returns an object type by monkeypatching the
# model classes to simple stand-ins.
def test_load_model_monkeypatched_classes(monkeypatch):
    class Dummy:
        def __init__(self, **kwargs):
            self.kw = kwargs

    # Patch the model classes in the module
    import utils.load_model as lm
    monkeypatch.setattr(lm, 'OpenAIModel', Dummy)
    monkeypatch.setattr(lm, 'AzureOpenAIModel', Dummy)
    monkeypatch.setattr(lm, 'HuggingFaceModel', Dummy)
    monkeypatch.setattr(lm, 'vllmModel', Dummy)

    cfg = {"class": "OpenAIModel", "model_name": "gpt-test", "api_key": "KEY"}
    model = load_model(cfg)
    assert isinstance(model, Dummy)
    assert model.kw['model_name'] == 'gpt-test'

    cfg = {"class": "AzureOpenAIModel", "model_name": "azure-model", "api_key": "KEY"}
    model = load_model(cfg)
    assert isinstance(model, Dummy)
    assert model.kw['model_name'] == 'azure-model'

    cfg = {"class": "HuggingFaceModel", "model_name": "hf-model", "device": "cpu"}
    model = load_model(cfg)
    assert isinstance(model, Dummy)
    assert model.kw['model_name'] == 'hf-model'

    cfg = {"class": "vllmModel", "model_name": "vllm-model"}
    model = load_model(cfg)
    assert isinstance(model, Dummy)
    assert model.kw['model_name'] == 'vllm-model'
