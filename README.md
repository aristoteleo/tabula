# **Tabula**

## **Preprint**
Toward a privacy-preserving predictive foundation model of single-cell transcriptomics with federated learning and tabular modeling

Jiayuan Ding, Jianhui Lin, Shiyu Jiang, Yixin Wang, Ziyang Miao, Zhaoyu Fang, Jiliang Tang, Min Li, Xiaojie Qiu

https://www.biorxiv.org/content/10.1101/2025.01.06.631427v1

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
├── README.md                  # Project description file
└── LICENSE  

```

## **Installation**
- CUDA >= 11.7
- Python >= 3.9
- flash-attn >= 2.3.5
- mpi4py >= 3.1.4
- Required dependencies are listed in [requirements.txt](requirements.txt)

Clone the repository:
```bash
$ git clone this-repo-url
$ cd tabula
```

Create your conda conda environment:
```bash
$ conda install -n tabula python=3.9
```
Install the torch:
```bash
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
Install dependencies:
```bash
$ pip install -r requirements.txt
```
To install mpi4py, follow these steps:
```bash
$ conda install mpi4py==3.1.4
```

To install flash-attention2, follow these steps, (For more information, check out [flash-attention](https://github.com/Dao-AILab/flash-attention)):
```bash
$ MAX_JOBS=4 pip install flash-attn==2.3.5 --no-build-isolation
```

## **Quick Start Tutorial**
Please see our example code in [tutorials](./tutorials/).
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
$ pip install pre-commit
$ pre-commit install
$ pre-commit run --all-files
```
Then, all future commits will call `black` automatically to format the code. Any code that does not follow the standard will cause a check to fail.

## **Contact**
For questions, feedback, or collaboration opportunities, please contact Xiaojie Qiu at xiaojie@stanford.edu and Jiayuan Ding at jiayuand@usc.edu.





