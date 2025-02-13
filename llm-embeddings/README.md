# Large Language Model Embeddings

This repository demonstrates a high-level workflow that combines modern NLP techniques with classical data analysis. It shows how to use a transformer-based model to generate text embeddings and then use them for downstream ML tasks.

## Getting Started

1. **Download the notebook:**
    Execute the following command to download the notebook file:
    ```
    wget https://raw.githubusercontent.com/PursuitOfDataScience/uchicago-workshops/refs/heads/main/llm-embeddings/llm-embeddings.ipynb
    ```

2. **Running the Script:**  
The notebook is designed to be executed on Midway3.  
    - **Step 1:** Request an interactive job. Either a CPU or GPU node is sufficient for this task. For further details, please refer to the user guide ([link](https://rcc-uchicago.github.io/user-guide/slurm/sinteractive/)).  
    - **Step 2:** Activate the `transformers` conda environment by executing:  
    ```
    module load python; source activate transformers;
    ```
    - **Step 3:** To run the Jupyter Notebook, please follow the instructions provided in this [link](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/?h=python).

3. **Miscellaneous:**  
If you do not have access to Midway3, the notebook can be executed locally or on [Google Colab](https://colab.research.google.com/). Please note that both the data and the model must be downloaded from Hugging Face. For further details, kindly consult the notebook.

Zoom link: https://uchicago.zoom.us/j/94809203196?pwd=bUAWrdLv9QfyWvTXFQ4otFiwGn8QGS.1