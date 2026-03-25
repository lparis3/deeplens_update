import numpy as np
import torch

from lenspop_NF_Architecture import build_joint_flow


def load_lenspop_flow(checkpoint_path="trained_Lenspop_flow.pt", device="cpu"):
    """
    Load the trained Lenspop normalizing flow and its normalization stats.
    """
    flow = build_joint_flow(device)
    flow.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    flow.load_state_dict(checkpoint["flow_state"])

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    return flow, mean, std


def sample_lenspop(flow, mean_data, std_data, n_samples, device="cpu"):
    """
    Sample (z_lens, z_source, sigma_v, theta_E) from the Lenspop flow.
    Returns physical values, with sigma_v dropped to match current pipeline usage.
    """
    with torch.no_grad():
        samples = flow.sample(n_samples).to(device)

    samples_array = samples.cpu().numpy()
    samples_physical = (samples_array * std_data) + mean_data
    samples_physical = np.delete(samples_physical, 2, axis=1)  # drop sigma_v
    return 10 ** samples_physical
