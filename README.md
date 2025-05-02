# STITCHING APP

### Installation

Supported version of Python >= 3.10 and <= 3.12 

Clone the repository
 ```bash
   git clone git@github.com:ikarus1211/StitcherA.git
   cd StitcherA
 ```  
Select one option to install environment
#### For Pip (Option 1)
Install a suitable version of PyTorch >= 2.3 and <= 2.6.0 
    ```
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    ```
Install requirements
    ```bash 
   pip install -r requirements.txt
    ```
#### For Conda (Option 2)
Install the environment ``` conda env create -f environment.yaml```

#### Light Glue
After you installed the environment using pip or conda, clone and install LightGlue repository. Don't forget to have the env activated
```bash
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

***Download the image data from:***

https://nextcloud.fit.vutbr.cz/s/wPxFXjBEbSFioQ6

### Run

To run simple example navigate to root folder and run:
```bash
python register.py --config-name map1.yaml
```
The configs are located in config folder. The adjustment can be made straight in config or by passing additional arguments:
```bash
python register.py --config-name map1.yaml data.final_res=[6400, 8400]
```