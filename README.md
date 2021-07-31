[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)


# dMelodies Controllability Experiments

## About
This repository contains the source code for running the experiments accompanying our ISMIR 2021 paper on disentanglement versus controllability in the context of symbolic music generation using the [dMelodies](https://github.com/ashispati/dmelodies_dataset) dataset.

Please cite as follows if you are using the code in this repository in any manner.

> Ashis Pati, Alexander Lerch. "Is Disentanglement enough? On Latent Representations for Controllable Music Generation”, in Proc. of the 22nd International Society for Music Information Retrieval Conference (ISMIR) , Online, 2021.

```
@inproceedings{pati2020dmelodies,
  title={Is Disentanglement enough? On Latent Representations for Controllable Music Generation},
  author={Pati, Ashis and Lerch, Alexander},
  booktitle={22nd International Society for Music Information Retrieval Conference (ISMIR)},
  year={2021},
  address={Online},
}
```

 

## Configuration
* Clone this repository and `cd` into the root folder of this repository in a terminal window. Run the following commands to initialize the submodules for the *dMelodies* and *dSprites* datasets:
    ```
    git submodule init
    git submodule update
    ```
    Alternatively the `--recurse-submodules` flag can be used with the `git clone` command while cloning the repository to directly initialize the datasets.  

* Install `anaconda` or `miniconda` by following the instruction [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

* Create a new conda environment using the `enviroment.yml` file located in the root folder of this repository. The instructions for the same can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

* Activate the `dmelodies` environment using the following command:
    ```
    conda activate dmelodies
    ```

## Contents
The contents of this repository are as follows:
* `dmelodies_dataset`: submodule containing the dMelodies dataset along with pyTorch dataloader and other helper code
* `src`: contains all the source code related to the different model architectures and trainers
    * `dmelodiesvae`: model architecture and trainer for the dMelodiesCNN and RNN, also contains the FactorVAE model
    * `utils`: module with model and training utility classes and methods
* other scripts to train / test the models and generate plots

## Usage
TO BE UPDATED.

**Note**: To be able to run the training scripts, the `dmelodies_dataset` folder must be added to the `PYTHONPATH`. This can be done form the command line by adding `PYTHONPATH=dmelodies_dataset` before the `python` command. For example,
```
PYTHONPATH=dmelodies_dataset python script_train_dmelodies.py
```
Alternatively, for IDEs such as PyCharm, the required folder can be added using the instructions [here](https://stackoverflow.com/questions/28326362/pycharm-and-pythonpath).
