import hydra
from omegaconf import DictConfig, OmegaConf

class Constants:
    pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def load(cfg: DictConfig):
    # 动态将 cfg 中的键值对映射到 Constants 类上，兼容您的旧代码
    for key, value in cfg.items():
        setattr(Constants, key, value)

