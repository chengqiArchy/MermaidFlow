# MermaidFlow: Redefining Agentic Workflow Generation via Safety-Constrained Evolutionary Programming

Welcome to the MermaidFlow project! This project provides a method to automatically genearte workflow, plan in Mermaid field and run in python field. 

## News

- Updates (2025-06-13) MermaidFlow has been accepted to the **ICML 2025 Workshop on Multi-Agent Systems**!
- Updates (2025-05-30) Initial upload to arXiv (see [PDF](https://arxiv.org/abs/2505.22967)).

## Project Structure

- `MermaidFlow_Linter/`  
  Tools for checking and validating Mermaid flowcharts. https://github.com/ChengqiCodeTrip/MermaidFlow_Linter.git
- `config/`  
  Stores model and LLM configuration files (such as `config2.yaml`).
- `run.py`  
  Main program entry point. Supports command-line arguments for running optimization and testing workflows.
- `pyproject.toml`  
  Project dependencies and metadata management.

## Quick Start

1. **Recommended: Use `uv` to Create and Manage Your Python Environment**

   `uv` is a modern Python package and environment management tool. It helps you quickly create isolated development environments and efficiently install dependencies. This prevents dependency conflicts and ensures your project runs consistently across different machines.

   First, make sure you have `uv` installed. If not, you can install it with:

   ```bash
   curl -LsSf https://astral.sh/uv/0.7.7/install.sh | sh
   ```

   Next you should pull the submodules:
   ```bash
   git submodule init
   git submodule update --remote
   ```

   In addition, you need to install the Mermaid CLI (`mmdc`), which is used to render Mermaid flowchart code into images. You can follow the official documentation for installation instructions: https://github.com/mermaid-js/mermaid-cli

   Typically, you need to have Node.js installed first. Then, you can install Mermaid CLI with the following command:

   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```

   After installation, you can use the `mmdc` command in your terminal to generate Mermaid images.

   > **Tip:** Using a virtual environment keeps your dependencies isolated from your system environment and helps avoid package conflicts.

2. **Configure the LLM (Large Language Model)**

   Open the `config/config2.yaml` file and enter your API key and the required model information.  
   For example:

   ```yaml
   models:
     "gpt-4o-mini":
       api_key: "your_api_key_here"
       # other configuration options...
   ```

   > **Note:** The API key is your credential for accessing the LLM service. Please keep it safe and do not share it publicly.

3. **Run the Code**

   Enter the following command in your terminal to start the main program:

   ```bash
   uv run run.py --dataset GSM8K --opt_model_name gpt-4o-mini --max_rounds 20 --validation_rounds 2
   ```

   Here’s what each argument means:
   - `--dataset GSM8K`: Specifies the dataset to use (in this case, GSM8K).
   - `--opt_model_name gpt-4o-mini`: Selects the model name for optimization.
   - `--max_rounds 20`: Sets the maximum number of optimization rounds to 20.
   - `--validation_rounds 2`: Runs validation every 2 rounds.

> You should have weave account
## Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@article{zheng2025mermaidflow,
  title={MermaidFlow: Redefining Agentic Workflow Generation via Safety-Constrained Evolutionary Programming},
  author={Zheng, Chengqi and Chen, Jianda and Lyu, Yueming and Ng, Wen Zheng Terence and Zhang, Haopeng and Ong, Yew-Soon and Tsang, Ivor and Yin, Haiyan},
  journal={arXiv preprint arXiv:2505.22967},
  year={2025}
}
```
## Acknowledgement

Special thanks to the following repositories for their invaluable code and prompt.

Our prompt is partially adapted from [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow). Our code and operators are partially adapted from [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow).