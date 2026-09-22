import os
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

@dataclass
class ExperimentConfig:
    name: str = "default"
    tag: str = ""
    use_timestamp: bool = False
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    ###

    resume: Optional[str] = None
    ckpt_path: Optional[str] = None

    data: dict = field(default_factory=dict)
    model_pl: dict = field(default_factory=dict)

    trainer: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
    checkpoint_epoch: Optional[dict] = None
    wandb: dict = field(default_factory=dict)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg