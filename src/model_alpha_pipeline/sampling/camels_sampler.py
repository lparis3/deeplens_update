import numpy as np
import torch
from model_alpha_pipeline.sampling.Camels_NF_Architecture import build_conditional_flow


def load_camels_flow(checkpoint_path="checkpoints/trained_Camels_flow.pt", device="cpu"):
    """
    Load the trained conditional Camels normalizing flow and normalization stats.
    """
    flow = build_conditional_flow(device)
    flow.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    flow.load_state_dict(checkpoint["flow_state"])

    condition_mean = checkpoint["condition_mean"]
    condition_std = checkpoint["condition_std"]
    dependent_mean = checkpoint["dependent_mean"]
    dependent_std = checkpoint["dependent_std"]

    return flow, condition_mean, condition_std, dependent_mean, dependent_std


def sample_camels(
    flow,
    condition_mean,
    condition_std,
    dependent_mean,
    dependent_std,
    log10_m_host_10sm,
    n_samples,
    device="cpu",
):
    """
    Sample max subhalo mass conditioned on host mass.

    Parameters
    ----------
    log10_m_host_10sm : float
        Host mass in units of log10(10^10 solar masses).
    """
    log10_m_host_normalized = (log10_m_host_10sm - condition_mean) / condition_std

    context = torch.tensor([[log10_m_host_normalized]], dtype=torch.float32, device="cpu")
    with torch.no_grad():
        samples = flow.sample(n_samples, context=context).to(device)

    samples_array = samples.cpu().numpy()
    log10_samples_physical = (samples_array * dependent_std) + dependent_mean
    samples_physical = 10 ** log10_samples_physical
    return samples_physical
