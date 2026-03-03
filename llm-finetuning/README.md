# LLM Supervised Fine-Tuning (SFT) Workshop

This repository contains a hands-on workshop notebook for fine-tuning large language models using **Full Fine-Tuning** and **LoRA (Low-Rank Adaptation)**.

**Model:** [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B)  
**Datasets:** [GAIR/LIMA](https://huggingface.co/datasets/GAIR/lima), [HuggingFaceH4/UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

## Getting Started

1. **Download the workshop files:**
    ```bash
    wget https://github.com/PursuitOfDataScience/uchicago-workshops/blob/main/llm-finetuning/llm-finetuning.ipynb
    wget https://github.com/PursuitOfDataScience/uchicago-workshops/blob/main/llm-finetuning/sft_utils.py
    ```

2. **Running the notebook on Midway3:**

    - **Step 1:** Request an interactive job with 1 H100 GPU (80G memory). For details, see the [user guide](https://rcc-uchicago.github.io/user-guide/slurm/sinteractive/).
    - **Step 2:** Load the `transformers` conda environment:
      ```bash
      module load python/anaconda-2022.05
      source activate transformers
      ```
    - **Step 3:** Launch Jupyter and open `llm-finetuning.ipynb`. Follow the instructions in this [link](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/?h=python) for running Jupyter Notebooks on Midway3.

    Alternatively, submit as a batch job by saving the following as `run.sh` and running `sbatch run.sh`:
    ```bash
    #!/bin/bash
    #SBATCH --job-name=llm-finetuning
    #SBATCH --account=<your-account>
    #SBATCH --partition=gpu
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:1
    #SBATCH --mem-per-gpu=80G
    #SBATCH --cpus-per-task=6
    #SBATCH --time=4:00:00
    #SBATCH --output=llm-finetuning.out
    #SBATCH --error=llm-finetuning.err

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    module load python/anaconda-2022.05
    source activate transformers

    jupyter nbconvert --to notebook --execute --inplace llm-finetuning.ipynb
    ```

3. **Miscellaneous:**  
    If you do not have access to Midway3, the notebook can be executed locally or on [Google Colab](https://colab.research.google.com/) with a GPU runtime. You will need to install the dependencies:
    ```bash
    pip install torch transformers datasets peft accelerate matplotlib
    ```
