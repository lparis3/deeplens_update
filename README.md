# deeplens_update

deeplens_update is an updated simulation pipeline for the deeplens ML4SCI collaboration. Functionally, deeplens_update is a wrapper package for lenstronomy and pyhalo that incorporates real astronomical images and utilizes a physically accurate sampling of galaxy and halo parameters. 

Using the pipeline will require the GalaxiesML dataset. The paper explaining the dataset can be found [here](https://arxiv.org/abs/2410.00271v1), and the dataset itself can be found [here](https://zenodo.org/records/11117528). 

The core dependencies for the pipeline are `numpy`, `scipy`, `matplotlib`, `tqdm`, `torch`, `nflows`,`lenstronomy`, `pyHalo`, `colossus`, `h5py`, `pandas`, `pyyaml`, `scikit-learn`, `seaborn`.

Please refer to the `new_pipeline_tutorial.ipynb` file for more specific instructions on installation, setup, and usage.
