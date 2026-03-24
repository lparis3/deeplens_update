import h5py
from lensing_sim.simulation.pipeline import simulation_parent

if __name__ == '__main__':
    cfg = {
        'lenspop': 'data/flows/trained_Lenspop_flow.pt',
        'camels': 'data/flows/trained_Camels_flow.pt'
    }

    instruments = ['LSST']
    DM_Types = ['SIDM']
    sim_number = 100

    hsc_data = h5py.File('data/deeplens_updated.hdf5', 'r')

    simulation_parent(DM_Types, instruments, sim_number, hsc_data, cfg)

