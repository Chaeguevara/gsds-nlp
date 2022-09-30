#!/usr/bin/env bash

conda create -n a2q3 python=3.7
conda activate a2q3

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install numpy