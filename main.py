import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from src.solver_disc import Solver_disc
from src.solver_pred import Solver_pred
from src.utils.general import save_yaml


def set_seeds(seed: int):
    """
    Set all the seeds for reproducible results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    """Parse CLI arguments.
    Returns:
        (argparse.Namespace, list): returns known and unknown parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cfg", type=str, default="configs/base.yml")
    parser.add_argument("--data-cfg", type=str, default="configs/datasets/iraven.yml")
    parser.add_argument("--model-cfg", type=str)
    return parser.parse_known_args()


def save_yaml_safe(path, cfg):
    """Safely save omegaconf.
    If the config already exists in the path, check that it's coherent with the
    current config (ensure consistence when re-running experiments). Otherwise
    creates it.
    Args:
        cfg (OmegaConf): config to be saved
        path (st): save path
    """
    if os.path.exists(path):
        existing_cfg = OmegaConf.load(path)
        assert sorted(existing_cfg) == sorted(cfg), (
            f"Found inconsistent omegaconf in {path}!"
            "Create a new experiment or check the current config."
        )
    else:
        save_yaml(path, OmegaConf.to_yaml(cfg, resolve=True))


def main():

    # parse pure CLI args
    args, unknown = parse_args()

    # parse CLI and yaml Omega args
    cfg_cli = OmegaConf.from_dotlist(unknown)
    cfg_base = OmegaConf.load(args.base_cfg)
    cfg_data = OmegaConf.load(args.data_cfg)
    cfg_model = OmegaConf.load(args.model_cfg)
    # merge order is important here, determines the hierarchy
    cfg = OmegaConf.merge(cfg_base, cfg_data, cfg_model, cfg_cli)

    set_seeds(cfg.seed)

    # add device cfg
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # get run name (name of the yaml file)
    run_name = os.path.basename(args.model_cfg).split(".")[0]

    if cfg.model.gridsearch:
        temp_list = cfg.model.temperature
        return_list = cfg.model.nreturn
        incontex_list = cfg.model.incontext
        for cfg.model.temperature in temp_list:
            for cfg.model.nreturn in return_list:
                for cfg.model.incontext in incontex_list:

                    # compose the full save path
                    cfg.path.full = os.path.join(
                        cfg.path.base,
                        cfg.data.dataset,
                        cfg.model.name,
                        run_name
                        + "tmp{}_nreturn{}_icl{}".format(
                            cfg.model.temperature,
                            cfg.model.nreturn,
                            cfg.model.incontext,
                        ),
                    )
                    cfg.model.path = cfg.path.full
                    os.makedirs(cfg.path.full, exist_ok=True)

                    # copy config file
                    copied_yml = os.path.join(cfg.path.full, "cfg.yml")
                    save_yaml_safe(copied_yml, cfg)

                    if cfg.model.classmode == "pred":
                        solver = Solver_pred
                    elif cfg.model.classmode == "disc":
                        solver = Solver_disc

                    s = solver(cfg)
                    s()
                    del s
    else:
        # compose the full save path
        cfg.path.full = os.path.join(
            cfg.path.base,
            cfg.data.dataset,
            cfg.model.name,
            run_name,
        )
        cfg.model.path = cfg.path.full
        os.makedirs(cfg.path.full, exist_ok=True)

        # copy config file
        copied_yml = os.path.join(cfg.path.full, "cfg.yml")
        save_yaml_safe(copied_yml, cfg)

        if cfg.model.classmode == "pred":
            solver = Solver_pred
        elif cfg.model.classmode == "disc":
            solver = Solver_disc

        s = solver(cfg)
        s()
    return


if __name__ == "__main__":
    main()
