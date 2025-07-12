# MermaidFlow: Redefining Agentic Workflow Generation via Safety-Constrained Evolutionary Programming

Welcome to the MermaidFlow project! This project provides a method to automatically genearte workflow, plan in Mermaid field and run in python field. 

## Project Structure

- `MermaidFlow_Linter/`  
  Tools for checking and validating Mermaid flowcharts.
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

   Next, you should pull wave first then create environment directly from uv:

   ```bash
   git clone https://github.com/ChengqiCodeTrip/weave_version_for_mermaidflow.git
   uv venv --python=python3.10
   uv run
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