# STITCHING APP

### Installation

Supported version of Python >= 3.10 and <= 3.12 

Clone the repository
 ```bash
   git clone git@github.com:ikarus1211/StitcherA.git
   cd StitcherA
 ```  
#### For Pip 
Install a suitable version of PyTorch >= 2.3 and <= 2.6.0 
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    ```
Install requirements
    ```bash 
   pip install -r requirements.txt
    ```
#### For Conda
Install the environment ```bash conda env create -f environment.yaml```

#### Light Glue
Clone and install LightGlue repository, don't forget to have the env activated
```bash
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

### Run

To run simple example navigate to root folder and run:
```bash
python register.py --config-name map1.yaml
```
The configs are located in config folder. The adjustment can be made straight in config or by passing additional arguments:
```bash
python register.py --config-name map1.yaml data.final_res=[6400, 8400]
```