# Trend Following

A quantitative financial research project focused on analyzing and replicating Trend-Following Investment strategies across historical daily futures data.

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) to manage the Python virtual environment and dependencies efficiently. The environment is defined in `pyproject.toml` and pinned in `uv.lock`.

### 1. Install `uv`
If you do not have `uv` installed on your system, you can install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*Or on macOS via Homebrew:*
```bash
brew install uv
```

### 2. Set Up the Environment
From the root of this repository, run the exact sync command:
```bash
uv sync
```
*What this does:*
- Detects or creates a `.venv` directory.
- Reads `uv.lock` to install the exact dependency versions required for the notebooks.
- Resolves the execution environment in milliseconds.

### 3. Adding New Dependencies
If you need to add a new package to the project, use the `uv add` command rather than installing via pip directly. This ensures `pyproject.toml` and `uv.lock` stay explicitly up-to-date:
```bash
uv add <package_name>
```

### 4. Launch the Project
To launch Jupyter Lab securely within the managed virtual environment, use the `uv run` command.

```bash
uv run jupyter lab
```

This ensures that the notebook kernels correctly inherit all dependencies (pandas, numpy, scipy, etc.) isolated to this project.

### 5. Data Requirements
For the notebooks to execute successfully, you must place the correct futures data file (`csi_data.duckdb`) into the `data/` directory of this repository.

> **Note:** Because this data is proprietary, it is *not* checked into Git. If you do not have `csi_data.duckdb` locally, please ask someone who works on this project for a copy. If you do not know who has it, please use your deductive reasoning skills to track it down!

### 6. Working with Notebooks (Jupytext)
This repository uses `jupytext` to pair Jupyter Notebooks (`.ipynb`) with plain Python scripts (`.py`). This makes version control much cleaner and allows you to edit the analysis logic in your favorite IDE.

Whenever you make changes to a Python script (e.g. `analysis.py` or `acf.py`), you must sync those changes back to the notebook before running the notebook server or committing:
```bash
uv run jupytext --sync <notebook>.ipynb
```

Conversely, if you make changes in the Jupyter Notebook interface, the sync command will update the paired `.py` script to match.
