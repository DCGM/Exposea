# EXPOSEA
is a standalone application designed for composing high-resolution images of large-scale documents.
It requires two types of input:

- A captured image of the entire document, referred to as the _reference_ image.
- Multiple close-up images capturing fine details, referred to as _fragments_.

The process begins by estimating a coarse homography between each fragment and the reference image.
Subsequently, a dense optical flow field is computed to achieve a more precise alignment between the pairs.
Finally, a light optimization is performed with respect to the reference image to ensure consistent color across composed
image form fragments.
 
### Installation

Supported version of Python >= 3.10 and <= 3.12 

Clone the repository
 ```bash
  git clone git@github.com:DCGM/Exposea.git
  cd Exposea
 ```  
Instalation is supported through pip and as a docker image
#### For Pip
Install requirements
```bash 
  pip install -r requirements.txt
```

#### For Docker

All files required to set up the Docker environment are located in the docker directory.

Start by modifying paths to input and output directories in docker compose

Afterwards you can build the docker as
```bash
  cd docker
  docker compose -f docker-compose.yaml build
```
### Run

#### Docker
You can now either run the _one-shot_ processing mode, which processes every task in the input directory once,
or start continuous mode, which keeps running and monitoring for new tasks.

**One-shot**
```bash
  docker compose up stitcher
```
**Continuous service**
```bash
  docker compose up stitcher_service
```
#### Python script
To run simple example as python script
```bash
  python python register.py  -i ./path/to/task -o ./path/to/output/
```
