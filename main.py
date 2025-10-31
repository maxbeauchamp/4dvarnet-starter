import hydra
from omegaconf import DictConfig, OmegaConf

# ⚠️ IMPORTANT: Import resolvers BEFORE anything else
from src import resolvers  # This auto-registers the resolvers

@hydra.main(config_path='config', config_name='main', version_base='1.3')
def main(cfg: DictConfig):
    """Main entrypoint."""
    
    # Debug: verify resolver works
    print("\n" + "="*80)
    print("CONFIGURATION LOADED")
    print("="*80)
    
    # Test that computed values work
    if "computed" in cfg:
        print(f"Computed input_vars_satellite: {cfg.computed.input_vars_satellite}")
        print(f"Computed input_vars_all: {cfg.computed.input_vars_all}")
    
    print("="*80 + "\n")
    
    # Run entrypoints
    for ep in cfg.entrypoints:
        hydra.utils.call(ep)

if __name__ == '__main__':
    main()

