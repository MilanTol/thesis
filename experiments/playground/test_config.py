from clumpy_halo_model import Config

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg = Config.from_yaml(root/"experiments/playground/config.yaml")
print(cfg.p("gas_profile"))
