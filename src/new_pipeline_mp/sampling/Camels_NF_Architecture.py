#Architecture used for conditional M_Host -> M_sub_max NF 
#%%
import torch
from torch import nn
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

# -------------------------
# Context MLP for coupling
# -------------------------
Num_Layers = 5
Hidden_Features = 64
context_dim = 1
D = 1  # dependent variable

def create_transform(num_layers=5, hidden_features=64):
    transforms = []
    for _ in range(num_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=D,
                hidden_features=hidden_features,
                context_features=context_dim,
            )
        )
    return CompositeTransform(transforms)


# -------------------------
# Flow builder
# -------------------------
def build_conditional_flow(device,context_dimension = context_dim, num_layers=Num_Layers, hidden_features=Hidden_Features):
    transform = create_transform(num_layers=num_layers,
                                 hidden_features=hidden_features)
    base_distribution = ConditionalDiagonalNormal(shape=[D],context_encoder=nn.Linear(context_dimension, 2))
    flow = Flow(transform, base_distribution).to(device)
    return flow

