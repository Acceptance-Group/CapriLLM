import yaml

from config import ModelConfig, TrainConfig
from tokenizer.core import BPETokenizer
from train import Trainer
from utils import count_parameters, format_number


def load_config(path: str, overrides=None):
    with open(path) as f:
        data = yaml.safe_load(f)
    model_config = ModelConfig(**data["model"])
    train_config = TrainConfig(**data["train"])
    for override in overrides or []:
        key, value = override.split("=", 1)
        section, field = key.split(".", 1)
        if section == "model":
            setattr(model_config, field, yaml.safe_load(value))
        else:
            setattr(train_config, field, yaml.safe_load(value))
    return model_config, train_config


def main(args):
    model_config, train_config = load_config(args.config, args.override)
    tokenizer = BPETokenizer.load(train_config.tokenizer_path)
    trainer = Trainer(model_config, train_config, tokenizer)
    params = count_parameters(trainer.model)
    print(f"Model parameters: {format_number(params)} ({params:,})")
    trainer.train()
