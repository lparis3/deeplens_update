import torch
from lensing_sim.flows.camels_flow import build_conditional_flow


def load_camels_flow(path, device="cpu"):
    flow = build_conditional_flow(device)
    checkpoint = torch.load(path, map_location=device)
    flow.load_state_dict(checkpoint['flow_state'])
    return flow, checkpoint


def sample_camels(flow, checkpoint, log10_M_host, n_samples):
    cm, cs = checkpoint['condition_mean'], checkpoint['condition_std']
    dm, ds = checkpoint['dependent_mean'], checkpoint['dependent_std']

    norm = (log10_M_host - cm) / cs
    context = torch.tensor([[norm]], dtype=torch.float32)

    with torch.no_grad():
        samples = flow.sample(n_samples, context=context).cpu().numpy()

    samples = (samples * ds) + dm
    return 10 ** samples
