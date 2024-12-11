# FuzzDistill ML

FuzzDistillML is a machine learning module that ingests data from FuzzDistillCC and generates models for FuzzDistillWeb.

> This module constitutes one-third of the FuzzDistill project. For further information on other modules and the project paper, please refer to the GitHub repository at [FuzzDistill](https://github.com/Saket-Upadhyay/FuzzDistill).

### Project Structure

```text
.
├── README.md
├── Training
│   ├── data
│   ├── final_training_scripts
│   │   ├── finalMLTrainerBB.py
│   │   ├── finalMLTrainerFN.py
│   │   ├── finalTFTrainerBB.py
│   │   └── finalTFTrainerFN.py
│   ├── tensorflowTrainerBB.ipynb
│   ├── tensorflowTrainerFN.ipynb
│   ├── testmultiplemodelsBB.ipynb
│   ├── testmultiplemodelsFN.ipynb
│   ├── trainallTF.py
│   ├── trainallXGB.py
│   ├── tuneHyperTFforFN.ipynb
│   └── xgboostFN.ipynb
├── includes
│   ├── constants.py
│   └── helpers.py
├── models
│   ├── BBpredict_TF_29_108_69.keras
│   ├── FNpredict_TF_21_32_86.keras
│   ├── XGBBasicBlockModel.pickle
│   └── XGBFunctionModel.pickle
└── requirements.txt
```
| Directory/File | Description                                                                                                                                                   |
|-----|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Training`    | Contains training scripts                                                                                                                                     |
|`includes`| Contains common function shared accross scripts                                                                                                               |
|`models`| Contains default pre-trained models discussed in the paper. New models from `trainall*.py` scripts stored here. `trainall*.py` will overrite existing models. |

## Training Notes
> **For detailed explanation of methods used, please read the paper at [FuzzDistill](https://github.com/Saket-Upadhyay/FuzzDistill).**

Prior to executing any notebook or script, it is imperative to have two essential components: Extracted Features from FuzzDistillCC and a Python 3 virtual environment.

### Dataset placement
Copy Extracted training features from FuzzDistillCC to `Training/data` directory. Default filenames are in `includes/constants.py`, change the following to suite your setup - 
```python
BB_DATASET_FILE = "data/bigBBFeatures.csv"
FN_DATASET_FILE = "data/bigFNFeatures.csv"
```
### Setup python env

```shell
cd ./FuzzDistillML
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

#### Tested on:
1. Ubuntu 22.04 (5.15.0-113-generic)
   * i9-13900KS
   * NVIDIA GeForce RTX 4080
2. macOS Sequoia 15.1.1
   * M3 Pro

  ---
  <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Saket-Upadhyay/FuzzDistillML">FuzzDistillML</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.linkedin.com/in/saketupadhyay/">Saket Upadhyay</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p>
