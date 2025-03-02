import json
import os

# Get the directory of the current script (main.py)
script_dir = os.path.dirname(os.path.realpath(__file__))
parameter_dir = os.path.normpath(os.path.join(script_dir, '../hyperparameters/'))

def load_parameters(json_name='0.json'):
    with open(f'{parameter_dir}/{json_name}') as f:
        parameters = json.load(f)
        
    return parameters
