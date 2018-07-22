# Defect-detection
Updated: March 20 2018

Loop defect detection packages using Faster-RCNN / SSD and local
image analysis methods.

Automated image analysis for open elliptical loop 
dislocation in STEM images of irradiated alloys


## Dependency
Python version >= 3.5 
* [ChainerCV](http://chainercv.readthedocs.io/en/latest/index.html)
* [Chainer](https://github.com/chainer/chainer)
* [OpenCV](https://opencv.org/)
* [Scikit-image](http://scikit-image.org/)

## Installation
Install miniconda (or Anaconda) for python 3

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
```
Download ChainerCV and go to the root directory of ChainerCV
```
git clone https://github.com/chainer/chainercv
cd chainercv
conda env create -f environment.yml
source activate chainercv
```
Install ChainerCV
```
pip install -e .
```
or
```
python setup.py install
```
Install Chainer (Only if you need to train the model)
```
pip install -U setuptools
pip install chainer
```
>>> **Enable CUDA/cuDNN support**

>>> In order to enable CUDA support, you have to install CuPy manually. If you also want to use cuDNN, you have to install CuPy with cuDNN support. See CuPy’s installation guide to install CuPy. Once CuPy is correctly set up, Chainer will automatically enable CUDA support.

Install scikit-images
```
conda install scikit-image
```
Download the defect detection code package and go to the directory of defect detection
```
cd ..
git clone https://github.com/leewaymay/defect-detection.git
cd defect-detection
```
## Usage

### Data
Download data of [LoopDefect](https://www.dropbox.com/sh/ttl5u14uzqxrili/AAAa1XMxP9AVJPQ3ie7xZZVxa?dl=0) and [MultiTypeDefect](https://www.dropbox.com/sh/yioyvrhy0yutwdm/AAA_RG84RphIvNtlEC4q7j1xa?dl=0). Remember the ```PATH``` of the data root directory. And use it when you create a dataloader with ```DefectDetectionDataset(data_dir= YOUR_DATA_PATH)```

### Train
To train the model on the dataset, run
```python train_faster_rCNN.py```

### Tutorial
See the tutorial of the project, see jupyter notebook ```Defect detection using faster-rCNN tutorial.ipynb```
