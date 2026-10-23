# PatchCore Defect Detection

A Streamlit GUI for industrial defect detection with PatchCore (via anomalib).
It covers importing and labeling images, building preprocessing pipelines,
training and validating models, and running parameter sweeps.

## Requirements

- Python 3.10 or newer
- Git
- Optional: an NVIDIA GPU with a recent driver, for faster training (CPU works too)

## Linux

Install:

```bash
git clone https://github.com/adamczykj95/image_processing.git
cd image_processing
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run:

```bash
source .venv/bin/activate
streamlit run image_processing/app/Home.py
```

If `python3 -m venv` fails on Debian/Ubuntu, install the venv package first:
`sudo apt install python3-venv`

## Windows (PowerShell)

Install Python 3.10+ from [python.org](https://www.python.org/downloads/) and
tick **"Add python.exe to PATH"** during setup.

Install:

```powershell
git clone https://github.com/adamczykj95/image_processing.git
cd image_processing
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Run:

```powershell
.venv\Scripts\Activate.ps1
streamlit run image_processing\app\Home.py
```

If paths cannot resolve properly, run powershell as administrator and then run:

```powershell
.venv\Scripts\Activate.ps1
streamlit run image_processing\app\Home.py
```

If activation fails with "running scripts is disabled on this system", run this
once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(In `cmd.exe`, activate with `.venv\Scripts\activate.bat` instead.)

### GPU on Windows

On Windows, pip installs a CPU-only build of PyTorch by default. To train on an
NVIDIA GPU, install a CUDA build of torch *before* running `pip install -e .`.
Pick the command for your setup at <https://pytorch.org/get-started/locally/>,
for example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Using the app

Streamlit opens <http://localhost:8501> in your browser. On the Home page,
create a new project or open an existing one. Put projects in a directory
outside this repo, such as `~/patchcore_projects/my_project`. Then work through
the sidebar pages in order:

1. Import Images
2. Label Images
3. Preprocessing
4. Train/Validate
5. Sweep

To stop the app, press Ctrl+C in the terminal.
