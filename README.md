# TUD-CSE-RP-RLinFinance

## Setup
### 1. Install poetry
You can install poetry from here: https://python-poetry.org/docs/#installing-with-the-official-installer
### 2. Verify installation
```
poetry --version
```
If poetry isn't recognized, you likely need to add it to PATH manually.
1. Copy the path to the folder containing `poetry.exe`
2. Add the folder to `PATH` via: `System Properties > Environment Variables > User > Path > Edit > New`

### 3. Download dependencies
(Optional but recommended) Use a `.venv` folder.
```
poetry config virtualenvs.in-project true
```
Create and install dependencies
```
poetry install
```
### 4. Using the virtual environment
Activate the virtual environment for the project:
```
poetry env activate
```
You can also run commands inside the environment without activating it by using:
```
poetry run <command>
``` 
You can add dependencies to the environment using:
```
poetry add <package-name>
```
For development dependencies
```
poetry add --dev <package-name>
``` 

<!-- Run `pip install -r requirements.txt` in the base folder of this project.

TA-Lib Note: If TA-Lib is part of the requirements, you might need to install the underlying TA-Lib C library first. Installation steps vary by operating system (Windows, macOS, Linux). Check the TA-Lib-wrapper documentation on PyPI or Conda-forge for instructions. stockstats (currently inside requirements.txt) is often used as an alternative within FinRL and doesn't have this external dependency. -->
