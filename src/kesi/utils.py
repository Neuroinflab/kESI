import os
import sys

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def write_run_summary(savedir, namespace=None):
    os.makedirs(savedir, exist_ok=True)
    launch_params = " ".join(sys.argv)

    savetext = "{} {}\n{}".format(sys.executable, launch_params, namespace)
    script_name = os.path.basename(sys.argv[0])
    with open(os.path.join(savedir, "run_summary {}.txt".format(script_name)), 'w') as f:
        f.write(savetext)
