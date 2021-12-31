#!/bin/bash

# utility script to quickly install everything
# required to run the evaluation scripts on an Ubuntu VM.

# run this script from where the requirements.txt file for the project is located

apt-get update
apt-get upgrade -y
apt-get install python3-pip virtualenv python3-opencv cmake -y
pip3 install dlib
pip3 install -r requirements.txt
