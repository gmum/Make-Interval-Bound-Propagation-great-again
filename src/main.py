import pyrootutils

from omegaconf import DictConfig
import hydra
from hydra.utils import call
import wandb

import utils


@hydra.main(version_base = None, config_path = "../configs", config_name = "config")
def main(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)
    call(config.exp.run_func, config)
    wandb.finish()

if __name__ == "__main__":
    pyrootutils.setup_root(
        search_from=__file__,
        indicator="requirements.txt",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )
    main()