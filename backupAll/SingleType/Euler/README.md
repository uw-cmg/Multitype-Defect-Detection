# Training in Euler with GPU

This file guides you to run GPU training in Euler.

## Preparation 
Some steps here may not needed if you have already done or used the 
alteraterive methods.

* Install the [Anaconda](https://www.anaconda.com/download/) .
* get [Wei's latest code](https://github.com/leewaymay/defect-detection)
* check you can use cuda 9.0 with `module avail` command
* download training data and remember the `PATH` of the data root directory.
* check all codes are in the **correct** directory that is accessible

## Submit the job

The script in the `Euler` folder can be used as the submitting scripts to 
submit jobs to Euler.

This script assumes you have already installed `chainerCV` if not you can 
uncomment the corresponding lines to install it.


