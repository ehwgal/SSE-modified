# SSE-modified  
_Thesis Project by Ellemijn Galjaard, under supervision of Nitya Tiwari and Shekhar Nayak_

## Credits
The code used for this thesis project is a modified version of the Self-Supervised Speech Enhancement model (SSE) by Yu-Che Jeffrey Wang, Shrikant Venkataramani, and Paris Smaragdis from the University of Illinois at Urbana-Champaign, Adobe Research (2020).  

- Link to their original code: https://github.com/jeffreyjeffreywang/SSE
- Link to their paper: https://arxiv.org/pdf/2006.10388.pdf

## Implemented Modifications
Some of the modifications we implemented include:  
- Changing the forward functions of the Encoder and Decoder in ```model.py``` to work with complex numbers.
- Have the STFT in ```stft.py``` return complex output.
- Have the first layer of the Encoder and last layer of the Decoder work with complex 3D input and 2D convolutions. We also added the parameters for the 2D convolutions (kernel_size, stride) in ```config.yaml```.
- Created a new evaluation script called ```test.py``` (based on code by Wang et al.).
- Introduced a new metric calculation in ```metrics.py``` (the NISQA function).

## Instructions for Running the Code
The code is specifically tailored to run on the GPU of the HPC cluster of the University of Groningen, called Hábrók.
Further documentation for this cluster can be found here: https://wiki.hpc.rug.nl/habrok/start  

As a first step, please go inside your scratch folder on the cluster (replace ```s0000000``` with your student number):  
```  
cd /scratch/s0000000/  

```
  
Then clone the repository:
```  
git clone https://github.com/ehwgal/SSE-modified.git  

```
(You can also clone via SSH by setting up an SSH key on the cluster via ssh-keygen, [as specified in the documentation](https://wiki.hpc.rug.nl/habrok/connecting_to_the_system/ssh_key_login?s[]=keygen)).  
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

Furthermore, if you haven't yet, please create [a Weights and Biases (WandB) account](https://wandb.ai/login) and [log in via the terminal](https://docs.wandb.ai/ref/cli/wandb-login) before running the jobs. This allows you to track the training via the wandb website.

### Data
After this, we need to obtain the data for training and testing.
The files used for data are too large to upload to GitHub, so please download them [from this Google Drive](https://drive.google.com/drive/u/0/folders/1QqpeEmKIfQERUHWt1yVtQSyGfB2GtEmf).
The ```urbansound16k.tar.gz``` is a pre-processed version of part of the UrbanSound8K data set [[1]](#1).
The ```librispeech_selection.tar.gz``` is a pre-processed version of the train-clean-100 subset of LibriSpeech [[2]](#2).

As the climbing gym noise data used in our project is not public, this code will simply show how to run the model with only the UrbanSound8K and LibriSpeech data as a demonstration, but results will not be significant. Please download both of the ```tar.gz``` files and place them in the main ```SSE-modified``` folder.  

> :warning: **DO NOT UNTAR THESE FILES. In the .sh scripts, these .tar.gz files are untarred to a local temporary directory.**
**After the job is done running, the temporary directory is deleted for more memory-efficient processing of data.**
**For more information on this, see: https://wiki.hpc.rug.nl/habrok/advanced_job_management/many_file_jobs**

### Instructions for training
After following the previous steps, we can start training the model.
The configuration settings for this model can be found in ```config.yaml```, this is where you can adjust things like the number of epochs or the batch size for training. Please make sure that the settings are to your liking.  

Then, go into the ```train_model.sh``` file and change the base_dir variable to your base_dir path (you should technically only have to change the student number here). In the terminal, this can be done using a vim or nano editor. Afterwards, please execute:
```
sbatch train_model.sh  

```

You can then check whether your model has started running by executing ```squeue -u <student number>``` in the command line. It will likely take a while for the job to start running, as we are making use of the long GPU node. Once the job has started running, it will output a slurm file that keeps track of the training progress. It is recommended to execute the following command:
```
tail -f <name of slurm file>

```
This will show you the progress in the terminal while the model is training. As seen in ```train.py```, the training loss is printed every 10 epochs, and a model checkpoint is outputted every 100 epochs. The final outputted model is called ```sep_trainer_final.pth```.

### Instructions for testing
For testing, we have added an additional metric to our ```metrics.py``` file called NISQA. In order to make this work, we first need to clone the NISQA project (as there is no pip library available yet). Please execute:
```
git clone https://github.com/gabrielmittag/NISQA.git

```
(Or do so via SSH). Make sure that the cloned NISQA project is right under the main ```SSE-modified``` folder of this project. 
After this, make sure you are logged into your wandb account as previously explained. 

Then, go into your ```test_model.sh``` file and change the base_dir variable (i.e. change the student number in the path).   
Now, you will be able to start testing:
```
sbatch test_model.sh

``` 

In ```test.py```, you can adjust all kinds of paremeter settings for the evaluation, such as the name of the trained model you want to evaluate. You can choose whether to create your own data set (which entails choosing an SNR and UrbanSound8K class to create the mixtures with), or you can choose to test on already available data (in this case, indicate the name of the ``.pkl``` file containing the data). Please have a look at these settings to see whether they are to your liking.


## References

[1] <a id="2"> Panayotov, V., Chen, G., Povey, D., & S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books" In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE. Aprl. 2015.</a>

[2] <a id="1">Salamon,J.,Jacoby,C. and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research" In 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.</a>
