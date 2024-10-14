class Trainers:
    def __init__(self):
        from .ppo_trainer import PPOTrainer
        from .sac_trainer import SACTrainer
        from .td3_trainer import TD3Trainer
        from .a2c_trainer import A2CTrainer
        from .amp_trainer import AMPTrainer
        from .ase_trainer import ASETrainer
        from .dtrans_trainer import DTransTrainer
        # from .hotu_trainer import HOTUTrainer
        from .physhoi_trainer import PhysHOITrainer

        self.trainer_map = {
            "ppo": PPOTrainer,
            "sac": SACTrainer,
            "td3": TD3Trainer,
            "a2c": A2CTrainer,
            "amp": AMPTrainer,
            "ase": ASETrainer,
            "dtrans": DTransTrainer,
            # "hotu": HOTUTrainer,
            "physhoi": PhysHOITrainer,
        }

    def __call__(self, trainer_name):
        return self.trainer_map[trainer_name]
