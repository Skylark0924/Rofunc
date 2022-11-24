from .franka_cabinet import FrankaCabinetTask
from .curi_cabinet import CURICabinetTask
from .curi_cabinet_bimanual import CURICabinetBimanualTask
from .curi_coffee_stirring import CURICoffeeStirringTask
from .humanoid import HumanoidTask
# from .humanoid_amp import HumanoidAMPTask

task_map = {
    "FrankaCabinet": FrankaCabinetTask,
    "CURICabinet": CURICabinetTask,
    "CURICabinetBimanual": CURICabinetBimanualTask,
    "CURICoffeeStirring": CURICoffeeStirringTask,
    "Humanoid": HumanoidTask,
    # "HumanoidAMPTask": HumanoidAMPTask,
}
