# Transfer Learning via Unsupervised Task Discovery <br/>for Visual Question Answering

## Requirements
* python2.7
* NVIDIA GPU with at least ?? GB memory
* At least ?? GB ram (for preloading all features into memory for faster learning)

### Setting with virtual environment

This code was tested under ubuntu 16.04 based on the following virtual environment setting.
We use a virtual environment with python 2.7.
```bash
virtualenv --system-site-packages -p python2.7 ~/venv_vqa_task_discovery
```
Activate the virtual environment with the command
```bash
source ~/venv_vqa_task_discovery/bin/activate
```
The python dependencies are installed by running the script
```bash
pip install -r requirements.txt
```
