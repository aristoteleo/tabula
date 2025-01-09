# **Tabula**

## **Overview**
A privacy-preserving predictive foundation model for single-cell transcriptomics, leveraging federated learning and tabular modeling.

## **Project Structure**
```plaintext
Tabula/  
├── resource/                      
│   ├── dataset/                   # Processed pretrian datasets  
│   ├── finetune_framework_x.yaml  # The configuration of downstream task  
│   ├── vocab.json                 # Genetic vocabulary

├── tabula/                      
│   ├── downstream/            # Downstream task implementations
│   ├── model/                 
│   │   ├── encoding/          # Single-cell data embedding  
│   │   ├── transformer/       # Transformer backbone  
│   ├── loss.py                # Training loss  

│   ├── training/              # Pre-training  
│   │   ├── config.py          # Configuration  
│   │   ├── data_loader.py     # Multi-client data loader  
│   │   ├── federater.py       # Federated framework  
│   │   ├── pretrainer.py      # PyTorch Lightning training framework  

├── tests/                     # Unit tests

├── tutorials/                 # Usage examples for downstream task

├── requirements.txt           # Python dependencies  
├──README.md                  # Project description file
└── LICENSE  

```

## **Requirements**
- CUDA >= 11.7
- Python >= 3.9
- flash-attn >= 2.3.5
- mpi4py >= 3.1.4
- Required dependencies are listed in [requirements.txt](requirements.txt).

Create your conda conda environment:
```bash
conda install -n tabula python=3.9
```
Install the torch version:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
Install dependencies:
```bash
pip install -r requirements.txt
```
To install mpi4py, follow these steps:
```bash
conda install mpi4py==3.1.4
```

To install mpi4py, follow these steps, (For more information, check out [flash-attention](https://github.com/Dao-AILab/flash-attention)):
```bash
MAX_JOBS=4 pip install flash-attn==2.3.5 --no-build-isolation
```

## **Quick Start Pretrain**
1. Download and preprocess your single-cell datasets and place them in the `resource/dataset` directory. The [demo dataset](1) can be downloaded.
2. Configure the all configuration of [framework.yaml](./tabula/framework.yaml)
3. Configure the script file, federated learning demo is [job.sh](./tabula/job.sh), There are 3 clients here, each occupying 1 GPU:
   ```bash
    #!/bin/bash
    #SBATCH --job-name=test
    #SBATCH --output=test.out
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=3
    #SBATCH --cpus-per-task=4
    #SBATCH --gpus=3

    # NCCL SETTING 
    export NCCL_BLOCKING_WAIT=1
    export NCCL_IB_PCI_RELAXED_ORDERING=1
    export NCCL_IB_RETRY_CNT=13

    # CUDA ENVIRONMENT 
    module load cuda/11.7

    # GET HOST ADDRESS
    srun hostname | sort -s | sort -n | sed 's/\..*$//' > hostsfile

    # MPIRUN 3 CLIENTS
    mpirun -np 3 -v -machinefile hostsfile python main.py

   ```
## **Quick Start Tutorial**

## Tabula Development Process
- Follow feature-staging-main review process
    - create a specific branch for new feature
    - implement and test on your branch; add unit tests
    - create pull request
    - discuss with lab members and merge into the main branch once all checks pass
- Follow python [Google code style](https://google.github.io/styleguide/pyguide.html)

## Code quality
- File and function docstrings should be written in [Google style](https://google.github.io/styleguide/pyguide.html)
- We use `black` to automatically format code in a standardized format. To ensure that any code changes are up to standard, use `pre-commit` as such.
```
# Run the following two lines ONCE.
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
Then, all future commits will call `black` automatically to format the code. Any code that does not follow the standard will cause a check to fail.

## **Contact**
## **License**





