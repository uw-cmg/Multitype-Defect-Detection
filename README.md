# multitype-defect-detection

This is the code written in [ChianerCV](https://chainercv.readthedocs.io/en/stable/) to do multitype defect detection for TEM images. And it is the code used for our paper [Multi Defect Detection and Analysis of Electron Microscopy 
Images with Deep Learning](https://doi.org/10.1016/j.commatsci.2021.110576). 

# Usuage

After install [ChianerCV](https://chainercv.readthedocs.io/en/stable/), you can train and test the codes with the following comands:

```bash

python train_multi_defect.py

```

To test the weights, you need to change the path of weight in the `test_multi_defect.py` to check the performance.

```bash

python test_multi_defect.py

```
