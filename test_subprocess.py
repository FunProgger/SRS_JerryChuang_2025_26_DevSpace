import subprocess
import os

# Simple test to verify subprocess calls within a conda environment
# Define the conda environment and path
result = subprocess.run(['python', '--version'], capture_output=True, text=True)
print(f"Python Version: {result.stdout.strip()}")

# Start the process and keep it open
# We open 'cmd.exe' as the base shell
process = subprocess.Popen(
    'cmd.exe', 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True, 
    shell=True
)

# Define your commands
commands = [
    r'call C:\Users\User\miniconda3\Scripts\activate.bat',
    'conda activate bivme311',
    'python --version',
    'echo Hello from the persistent session!'
]

# Send each command into the OPEN process
for cmd in commands:
    process.stdin.write(cmd + '\n')

# Important: You must eventually close it or tell it you are done
stdout, stderr = process.communicate() 
print(stdout)