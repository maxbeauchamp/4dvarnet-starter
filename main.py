import hydra
from omegaconf import DictConfig


@hydra.main(config_path='config', config_name='main', version_base='1.3')
def main(cfg: DictConfig):
    """Main entrypoint: run the entrypoints of the selected xp config."""
    for ep in cfg.entrypoints:
        hydra.utils.call(ep)


if __name__ == '__main__':
    main()
