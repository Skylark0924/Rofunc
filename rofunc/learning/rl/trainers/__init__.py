from .ppo_trainer import PPOTrainer
from .sac_trainer import SACTrainer
from .td3_trainer import TD3Trainer

trainer_map = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "td3": TD3Trainer
}
