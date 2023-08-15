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
module load ...
python -m venv .env  

```  

If you change the name of the environment, please be sure to change the name of the environment in the bash scripts as well. 


### Instructions for training


### Instructions for testing
