
import os
import sys
import json
import numpy as np

if __name__ == '__main__':
    progname = os.path.basename(sys.argv[0])

    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
        print(f'usage: {progname} file')
        sys.exit(0)

    fname = sys.argv[1]
    if not os.path.isfile(fname):
        print(f'{progname}: {fname}: no such file')
        sys.exit(1)

    data = np.load(fname, allow_pickle=True)
    if 'config' not in data:
        print(f"{progname}: file {fname} does not contain a 'config' key.")
        sys.exit(1)

    config = data['config']
    if isinstance(config, np.ndarray):
        config = config.item()
    print(json.dumps(config, indent=4))
    
