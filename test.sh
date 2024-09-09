#!/bin/bash

cd gpt-neo
python3.10 model.py 125M
python3.10 model.py 1.3B
python3.10 model.py 2.7B
cd ..
cd SmolLM
python3.10 model.py 135M
python3.10 model.py 360M
python3.10 model.py 1.7B
cd ..
cd tinyLLaMA
python3.10 model.py
cd ..
cd mobileLLaMA
python3.10 model.py
cd ..
cd Gemma
python3.10 model.py

