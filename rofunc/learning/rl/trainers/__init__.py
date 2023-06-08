from .ppo_trainer import PPOTrainer
from .sac_trainer import SACTrainer

trainer_map = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
}
