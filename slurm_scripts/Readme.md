# Demucs Training and Inference Workflow 

## 1. Create the Demucs Environment 

Open a terminal in the `project/demucs` folder and run:


```bash
conda env update -f environment-cuda.yml
conda activate demucs
pip install -e .
```


This creates and activates the Conda environment for Demucs and installs the project in editable mode so that any changes in the repository are reflected immediately.



---



## 2. Update the Conda Path in SLURM Scripts 

Before running any jobs, you must update the SLURM scripts with the correct path to your `conda.sh` initialization file. For convenience, we provide an update script that does this automatically.

### 2.1 Using the Update Script 


```bash
chmod +x update_conda_path.sh
```
 
6. **Run the script**  (from `project/demucs/scripts`) with your full Conda initialization path. For example:


```bash
./update_conda_path.sh /home/yourusername/miniconda3/etc/profile.d/conda.sh
```


This will replace the Conda activation line in every SLURM script with the provided path.


> **Note:**  The Demucs repository path is assumed to be fixed (you already include `cd ..` in the scripts) so you do not need to update it.



---



## 3. Workflow Commands 


Once your environment and SLURM scripts are set up, you can run the following steps:


### 3.1 Training an HDemucs Model 


Submit the training SLURM job (which runs a distributed training using Dora):



```bash
sbatch run_slurm.sh
```

This command will launch an 8-task distributed training job (each task with one GPU). The training command is embedded within the SLURM script (using `dora run -d` with your model hyperparameters). Check the output logs (in `slurm_out/`) for progress and error messages.

### 3.2 Export a Trained Model 

After training, export the model so that it is available for inference. Provide the model signature (e.g., `53288a8c` or your custom name). Submit the export job:


```bash
sbatch slurm_release.sh <model_name>
```

If no model name is provided, it defaults to `53288a8c`.

### 3.3 Testing/Evaluating the Model 


Evaluate the exported model using the test script by running:



```bash
sbatch evaluate_run_slurm.sh <model_name>
```

This script will use the command `python3 -m tools.test_pretrained --repo ./release_models -n <model_name>` and log results (SDR and other metrics) in the output files.

### 3.4 Separating a Specific Audio File 


To separate a particular song or audio file using your exported model, run:



```bash
sbatch slurm_separate.sh <file_path> <model_name>
```


For example:



```bash
sbatch slurm_separate.sh test_2.mp3 53288a8c
```

This command uses the Demucs command (`demucs --repo ./release_models -n <model_name> <file_path>`) to process the file and output separated stems into the default folder.

### 3.5 Continuing Training from a Checkpoint 


If you need to resume training (or fine-tune further) from an existing checkpoint, run:



```bash
sbatch continue_run_slurm.sh <model_name>
```

Replace `<model_name>` with the signature of the model you wish to continue training. This SLURM script calls `dora run -d -f <model_name>` to resume training from the last saved state.


---



## 4. Additional Information 


### Original Demucs Information and Training Docs 

For further technical details on the Demucs models, training hyperparameters, and dataset details, please refer to the original [README.md](https://github.com/facebookresearch/demucs/blob/main/README.md)  and [training.md](https://github.com/facebookresearch/demucs/blob/main/docs/training.md)  provided in the repository. They explain model architectures (e.g., HDemucs, HTDemucs), evaluation metrics (SDR, MOS), and other advanced usage. If you encounter issues or need to experiment with different configurations, consult the Hydra-based configuration system explained in the training documentation.

### Building the Soundstretch Binary 

Demucs uses `soundstretch` from the [SoundTouch library](https://www.surina.net/soundtouch/soundstretch.html)  for pitch/tempo augmentation. Although many systems provide a pre-built version, if you need to compile your own you can follow the instructions provided on the SoundTouch website. Ensure that the resulting executable is placed (or linked) in the appropriate directory so that your SLURM scripts can find it via the PATH. In our SLURM scripts, we add the soundstretch directory with:


```bash
export PATH="$(pwd)/soundtouch/build:$PATH"
```

This assumes that your `soundtouch` build directory is relative to the current repository root (with the SLURM scripts located in `project/demucs/scripts` and you `cd ..` to move to `project/demucs`).

### Environment Variables Recap 

 
- **CONDA_SH_PATH:**  This variable must be set to the full path of your `conda.sh` file. Use the update script to set this in all SLURM scripts.
 
- **DEMUCSPATH:**  Since your scripts are always run from `project/demucs/scripts`, they include a fixed command `cd ..` to move to the repository root. There is no need to update this as it is constant for your project.



---



## 5. Summary of Commands 


Below is a concise list of commands you (or your teacher) need to run:

 
2. **Create the Environment:** 


```bash
conda env update -f environment-cuda.yml
conda activate demucs
pip install -e .
```
 
4. **Update Conda Path in SLURM Scripts (from project/demucs/scripts):** 


```bash
python3 update_conda_path.py /your/full/path/to/conda.sh
```
 
6. **Train an HDemucs Model:** 


```bash
sbatch run_slurm.sh <number of epochs (390 in default)>
```
 
8. **Export a Trained Model:** 


```bash
sbatch slurm_release.sh <model_name>
```
 
10. **Test/Evaluate a Model:** 


```bash
sbatch evaluate_run_slurm.sh <model_name>
```
 
12. **Separate a Specific File:** 


```bash
sbatch slurm_separate.sh <file_path> <model_name>
```
 
14. **Continue Training from a Checkpoint:** 


```bash
sbatch continue_run_slurm.sh <model_name>
```



---



## 6. Final Notes 

 
- Make sure to adjust any SLURM-specific parameters (e.g., partition, account, memory, GPU requirements) to match your cluster configuration.
 
- Always run these commands from the appropriate directory (i.e., from within `project/demucs/scripts`, since the SLURM scripts assume that the repository root is one directory up).