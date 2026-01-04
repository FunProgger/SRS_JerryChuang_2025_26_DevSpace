"""
Automate cardiac volume computation using conda environment activation.

This script automates the execution of the compute_volume.py script from the
biv-me (biventricular modeling) package across multiple fitted cardiac model
directories. It handles conda environment activation and executes volume
calculations for each model directory found in the specified input folder.

Parameters
----------
CONDA_ACTIVATE : str
    Full path to conda's activate.bat script (e.g., 'C:\\Users\\User\\miniconda3\\Scripts\\activate.bat')
    Located in the same folder as Scripts\\conda.exe
CONDA_ENV : str
    Name of the conda environment to activate (default: 'bivme311')
OUTPUT_DIR : str
    Directory path containing the compute_volume.py script
    (e.g., 'C:\\Project\\SRS\\biv_me public\\biv-me\\src\\bivme\\analysis')
MODEL_DIR : str
    Root directory path containing fitted cardiac model subdirectories
    (e.g., 'C:\\Project\\SRS\\dev_space\\case')
CALC_DIR : str
    Directory path where computed volume results will be saved
    (e.g., 'C:\\Project\\SRS\\dev_space\\analysis')

Process
-------
1. Scans MODEL_DIR for subdirectories containing fitted cardiac models
2. Activates the specified conda environment using CONDA_ACTIVATE
3. Changes directory to OUTPUT_DIR where compute_volume.py is located
4. For each model directory:
   - Runs compute_volume.py with the following arguments:
     -mdir : Input fitted model path (MODEL_DIR/[dir_name])
     -p : Number of processes (set to 1)
     -o : Output directory path (CALC_DIR/[dir_name]_volumes)

Notes
-----
- Currently configured to process only the first 2 directories (range(2))
- Adjust the loop range to process all directories if needed
- The script uses subprocess.Popen to maintain an open cmd.exe session
- All commands are sent to the same process to preserve environment activation

Examples
--------
Directory structure:
    case/
    ├── patient1/
    │   └── [fitted model files]
    └── patient2/
        └── [fitted model files]

Output:
    analysis/
    ├── patient1_volumes/
    │   └── [computed volumes]
    └── patient2_volumes/
        └── [computed volumes]
"""
import subprocess
import os

# ### Test directory 

# locate conda active by using anaconda prompt and running "where conda"
# activate.bat should be in the same folder as Scripts\conda.exe
# CONDA_ACTIVATE = r''

# CONDA_ENV = 'bivme311'

# OUTPUT_DIR = r'C:\Project\SRS\biv_me public\biv-me\src\bivme\analysis'

# MODEL_DIR = r'C:\Project\SRS\dev_space\case'

# CALC_DIR = r"C:\Project\SRS\dev_space\analysis"

# ###

CONDA_ACTIVATE = r'C:\Users\User\miniconda3\Scripts\activate.bat'

CONDA_ENV = 'bivme311'

OUTPUT_DIR = r'C:\Project\SRS\biv_me public\biv-me\src\bivme\analysis'

MODEL_DIR = r'C:\Project\SRS\dev_space\case'

CALC_DIR = r"C:\Project\SRS\dev_space\analysis"

# Use a list comprehension to get only directory names
dir_names = [f.name for f in os.scandir(MODEL_DIR) if f.is_dir()]

for i in dir_names:
    print(i)

new_dir = os.path.join(OUTPUT_DIR, dir_names[0])
print(new_dir)

def run_compute_volume():
    # Define the conda environment and path

    # Start the process and keep it open
    process = subprocess.Popen(
        'cmd.exe', 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        shell=True
    )

    # Define your commands
    setup_env = [
        f'call {CONDA_ACTIVATE}',
        f'conda activate {CONDA_ENV}',
        f'cd {OUTPUT_DIR}'
    ]

    # Send each command into the OPEN process
    for setup in setup_env:
        process.stdin.write(setup + '\n')

    # TODO: currently set to only run first 2 directories for testing
    # Loop through directories and run compute_volume.py
    for i in range(len(dir_names)):
        command = f'python compute_volume.py -mdir {MODEL_DIR}/{dir_names[i]} -p 1 -o {CALC_DIR}/{dir_names[i]}_volumes'
        process.stdin.write(command + '\n')

    # Important: You must eventually close it or tell it you are done
    stdout, stderr = process.communicate() 
    print(stdout)

run_compute_volume()