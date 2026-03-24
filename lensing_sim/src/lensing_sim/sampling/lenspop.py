import torch
import numpy as np
from lensing_sim.flows.lenspop_flow import build_joint_flow


def load_lenspop_flow(path, device="cpu"):
    flow = build_joint_flow(device)
    checkpoint = torch.load(path, map_location=device)
    flow.load_state_dict(checkpoint['flow_state'])
    return flow, checkpoint['mean'], checkpoint['std']


def sample_lenspop(flow, mean, std, n_samples):
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    samples = (samples * std) + mean
    samples = np.delete(samples, 2, axis=1)

    return 10 ** samples
