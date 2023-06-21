from .ant import AntTask
from .cartpole import CartpoleTask
from .curi_cabinet import CURICabinetTask
from .curi_cabinet_bimanual import CURICabinetBimanualTask
from .curi_coffee_stirring import CURICoffeeStirringTask
from .franka_cabinet import FrankaCabinetTask
from .humanoid import HumanoidTask
from .humanoid_amp import HumanoidAMPTask
from .ase.humanoid_amp_getup import HumanoidAMPGetupTask

task_map = {
    "Ant": AntTask,
    "Cartpole": CartpoleTask,
    "FrankaCabinet": FrankaCabinetTask,
    "CURICabinet": CURICabinetTask,
    "CURICabinetBimanual": CURICabinetBimanualTask,
    "CURICoffeeStirring": CURICoffeeStirringTask,
    "Humanoid": HumanoidTask,
    "HumanoidAMP": HumanoidAMPTask,
    "HumanoidASEGetupSwordShield": HumanoidAMPGetupTask,
}
