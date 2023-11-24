# KeyFind

This script repo is designed to classify song keys using a pre-trained audio classification model from the Hugging Face library.

## Overview

The script performs audio classification to identify song keys. It leverages the powerful models available in the Hugging Face library, making it a robust tool for audio analysis in the domain of music.

## Prerequisites

Before running the script, ensure you have the following prerequisites installed:

- Python 3.11
- PyTorch 2.0.1
- Hugging Face's Transformers library 4.34.0 
- NumPy
- Librosa (for audio processing)

## Installation

Before running the script, you need to install the required dependencies. The script has been tested with specific versions of each package, so it's recommended to use these versions to ensure compatibility.

## Setup conda environment

If1 you don't have Conda installed, download and install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). After installation, follow these steps to set up your environment:

1. **Create a New Conda Environment**: Open the Anaconda Prompt and create a new Conda environment with Python 3.11:
    `conda create -n songkey_classifier python=3.11`
    
    Activate the environment:
    `conda activate songkey_classifier`
    
2. **Install PyTorch**: Install PyTorch 2.0.1. Make sure to select the version compatible with your Windows system and CUDA version (if you plan to use GPU acceleration) from the [PyTorch website](https://pytorch.org/get-started/locally/). Here's an example command:
    `conda install pytorch=2.0.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch`

    Replace `cudatoolkit=11.3` with the version that matches your CUDA installation, or omit it for CPU-only installation.
    
3. **Install Hugging Face's Transformers**: Install the Transformers library version 4.34.0:
    `pip install transformers==4.34.0`
    
4. **Install Additional Dependencies**: Install NumPy and Librosa for audio processing:
    `pip install numpy librosa`
    

### Verify Installation

After installation, you can verify that the correct versions of the packages are installed:

`python -c "import torch; print(torch.__version__)" python -c "import transformers; print(transformers.__version__)" python -c "import numpy; print(numpy.__version__)" python -c "import librosa; print(librosa.__version__)"`

You are now ready to run the script with all the necessary dependencies installed.

## Input Data
Download `encoded_dataset.parquet` from the link I sent you and place it inside the directory that the repository is located.

## Usage

To use the script, you need to provide the path to the pre-trained model as a command-line argument. The model should be compatible with the Hugging Face Transformers library and suitable for audio classification tasks.

Run the script using the following command:
`python exp3.py "facebook/hubert-large-ls960-ft"

