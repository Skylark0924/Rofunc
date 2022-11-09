from .franka_cabinet import FrankaCabinet
from .curi_cabinet import CURICabinet
from .curi_coffee_stirring import CURICoffeeStirring

task_map = {
    "FrankaCabinet": FrankaCabinet,
    "CURICabinet": CURICabinet,
    "CURICoffeeStirring": CURICoffeeStirring,
}
