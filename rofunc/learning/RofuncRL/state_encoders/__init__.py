from .base_encoders import BaseEncoder, MLPEncoder
from .visual_encoders import CNNEncoder, ResnetEncoder, ViTEncoder
from .graph_encoders import HomoGraphEncoder, HeteroGraphEncoder

encoder_map = {
    # Encoders for vector data
    'mlp': MLPEncoder,
    # Encoders for image data
    "cnn": CNNEncoder,
    "resnet": ResnetEncoder,
    "vit": ViTEncoder,
    # Encoders for graph data
    "homograph": HomoGraphEncoder,
    "heterograph": HeteroGraphEncoder,
    # Encoders for language data
    # "bert": BertEncoder,
    # "gpt": GPTEncoder,
    }
