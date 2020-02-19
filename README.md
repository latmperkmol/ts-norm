# ts-norm
Time series normalization for satellite imagery.

The paper corresponding to this work is published in the Journal of Computers and Electronics in Agriculture and [can be found here.](https://doi.org/10.1016/j.compag.2019.104893)

## Installation
ts-norm runs as a script, primarily using functions out of custom_utils.py. To "install", simply download the repo.

Creating an anaconda environment is highly recommended.
```
conda create --name tsnorm
conda activate tsnorm
conda install numpy scipy matplotlib seaborn scikit-image
conda install gdal
conda install geopandas rasterio
conda install -c conda-forge basemap pykridge  # optional, used for some arosics functions
pip install arosics  # also optional, used for image co-registration
```

## Usage
Currently, only the python interface is supported, but a CLI will be implemented.

Executing `custom_utils.main` will normalize a target image to a reference image. Note that a substantial number of intermediate products are currently written to the disk during this process, so making a new folder for your outputs is advisable. 

A Jupyter Notebook demonstrating a multi-sensor application is included. That is the best way to get familiar with the script!
