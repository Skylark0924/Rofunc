from .franka_cabinet import FrankaCabinet
from .curi_cabinet import CURICabinet

task_map = {
    "FrankaCabinet": FrankaCabinet,
    "CURICabinet": CURICabinet,
}
