from .ant import AntTask
from .cartpole import CartpoleTask
from .curi_cabinet import CURICabinetTask
from .curi_cabinet_bimanual import CURICabinetBimanualTask
from .curi_coffee_stirring import CURICoffeeStirringTask
from .franka_cabinet import FrankaCabinetTask
from .humanoid import HumanoidTask
from .humanoid_amp import HumanoidAMPTask
from .ase.humanoid_amp_getup import HumanoidAMPGetupTask
from .ase.humanoid_perturb import HumanoidPerturbTask
from .ase.humanoid_heading import HumanoidHeadingTask
from .ase.humanoid_location import HumanoidLocationTask
from .ase.humanoid_reach import HumanoidReachTask
from .ase.humanoid_strike import HumanoidStrikeTask


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
    "HumanoidASEPerturbSwordShield": HumanoidPerturbTask,
    "HumanoidASEHeadingSwordShield": HumanoidHeadingTask,
    "HumanoidASELocationSwordShield": HumanoidLocationTask,
    "HumanoidASEReachSwordShield": HumanoidReachTask,
    "HumanoidASEStrikeSwordShield": HumanoidStrikeTask,
}
