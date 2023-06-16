from .ppo_trainer import PPOTrainer
from .sac_trainer import SACTrainer
from .td3_trainer import TD3Trainer
from .a2c_trainer import A2CTrainer

trainer_map = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "td3": TD3Trainer,
    "a2c": A2CTrainer
}
