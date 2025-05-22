# Large Language Model Inference

This repository comprises two notebooks relevant to LLM inference.

## Getting Started

1. **Download the notebook:**
    Execute the following command to download the notebook files:
    ```
    wget https://github.com/PursuitOfDataScience/uchicago-workshops/blob/main/llm-inference/llm-inference.ipynb
    ```

    ```
    wget https://github.com/PursuitOfDataScience/uchicago-workshops/blob/main/llm-inference/vllm.ipynb
    ```

2. **Running the Script:**  
The notebook is designed to be executed on Midway3.  
    - **Step 1:** Request an interactive job. 1 A100 is sufficient for this task. For further details, please refer to the user guide ([link](https://rcc-uchicago.github.io/user-guide/slurm/sinteractive/)).  
    - **Step 2:** To run `llm-inference.ipynb`, activate the `pytorch` conda environment by executing:  
    ```
    module load python; source activate pytorch;
    ```
    To run `vllm.ipynb`, activate the `vllm_serving` conda environment by executing:
    ```
    module load python; source activate vllm_serving;
    ```
    - **Step 3:** To run the Jupyter Notebook, please follow the instructions provided in this [link](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/?h=python).

3. **Miscellaneous:**  
If you do not have access to Midway3, the notebook can be executed locally or on [Google Colab](https://colab.research.google.com/).

