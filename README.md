# SSE-modified  
_Thesis Project by Ellemijn Galjaard, under supervision of Nitya Tiwari and Shekhar Nayak_

## Credits
The code used for this thesis project is a modified version of the Self-Supervised Speech Enhancement model (SSE) by Yu-Che Jeffrey Wang, Shrikant Venkataramani, and Paris Smaragdis from the University of Illinois at Urbana-Champaign, Adobe Research (2020).  

- Link to their original code: https://github.com/jeffreyjeffreywang/SSE
- Link to their paper: https://arxiv.org/pdf/2006.10388.pdf

## Instructions for Running the Code
The code is specifically tailored to run on the GPU of the HPC cluster of the University of Groningen, called Hábrók.
Further documentation for this cluster can be found here: https://wiki.hpc.rug.nl/habrok/start  

As a first step, please go inside your scratch folder on the cluster and replace ```s0000000``` with your student number:  
```  
cd /scratch/s0000000/  

```
  
Then clone the repository:
```  
git clone https://github.com/ehwgal/SSE-modified.git  

```
(You can also clone via SSH by setting up an SSH key on the cluster via ssh-keygen, see the documentation: https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/ssh_key_login?s[]=keygen).  
Once cloning is succesful, go inside the cloned repo:
``` 
cd SSE-modified  

```

Before training or testing with the code, please create a virtual environment. We will create an environment that uses Python version 3.10.4 here.
In the terminal, run:  
```
module purge
module load Python/3.10.4-GCCcore-11.3.0-bare
python -m venv .env  

```  

The .sh scripts will automatically activate the environment and install the required dependencies that can be found in ```requirements.txt```. If you change the name of the environment, please be sure to change the name of the environment in the .sh files as well. 

After this, we need to obtain the data for training and testing.
The files used for data are too large to upload to GitHub, so please download them here:
The ```urbansound16k.tar.gz``` is a pre-processed version of part of the UrbanSound8K data set (CITE).
The ```librispeech_selection.tar.gz``` is a pre-processed version of the train-clean-100 subset of LibriSpeech (CITE).
The folder additionally contains a ```BASELINE.pth``` file which can be used for trying out the test script, in case you do not want to train a model from scratch.

As the climbing gym noise data used in our project is not public, this code will simply show how to run the model with only the UrbanSound8K and LibriSpeech data. Please download both of the ```tar.gz``` files and place them in the main ```SSE-modified``` folder.
**DO NOT UNTAR THESE FILES. In the .sh scripts, these data files are untarred to a local temporary directory.**
**After the job is done running, the temporary directory is deleted for more memory-efficient processing of data.**
**For more information on this, see: ...**

### Instructions for training
After following the previous steps, we can start training the model.
The configuration settings for this model can be found in config.yaml

### Instructions for testing
