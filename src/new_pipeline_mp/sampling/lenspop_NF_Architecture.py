#Architecture used for (zlens,zsrc,sigmam_v,theta_E) NF 
#%%
import torch
from torch import nn
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import RandomPermutation

# Optional: ActNorm if installed
try:
    from nflows.transforms.normalization import ActNorm
    HAS_ACTNORM = True
except ImportError:
    HAS_ACTNORM = False

# -------------------------
# Context MLP for coupling
# -------------------------

Num_Layers = 12
Hidden_Features = 256

class ContextMLP(nn.Module):
    """MLP that ignores optional context (to match nflows coupling API)."""
    def __init__(self, in_features, out_features, hidden_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
    def forward(self, x, context=None):
        return self.net(x)

def make_net(in_features, out_features, hidden_features=Hidden_Features):
    return ContextMLP(in_features, out_features, hidden_features)

def create_transform(num_layers=Num_Layers, hidden_features=Hidden_Features):
    transforms = []
    D = 4  # (zl, zs, sigma,theta_E)

    if HAS_ACTNORM:
        transforms.append(ActNorm(features=D))

    # Alternating binary masks for #D
    mask_a = torch.tensor([True, True, True, False])   
    mask_b = torch.tensor([True, True, False,True]) 
    mask_c = torch.tensor([True, False,True,True])
    mask_d = torch.tensor([False, True, True, True])   
    masks = [mask_a,mask_b,mask_c,mask_d] * 4

    for i in range(num_layers):
        mask = masks[i]
        transforms.append(
            AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_f, out_f: make_net(in_f, out_f, hidden_features)
            )
        )
        transforms.append(RandomPermutation(features=D))
        if HAS_ACTNORM:
            transforms.append(ActNorm(features=D))

    return CompositeTransform(transforms)

# -------------------------
# Flow builder
# -------------------------
def build_joint_flow(device, num_layers=Num_Layers, hidden_features=Hidden_Features):
    transform = create_transform(num_layers=num_layers,
                                 hidden_features=hidden_features)
    base_distribution = StandardNormal(shape=[4])
    flow = Flow(transform, base_distribution).to(device)
    return flow



