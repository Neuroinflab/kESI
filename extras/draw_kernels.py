import argparse
import json
import os.path

import numpy as np
import pylab as pb

def main():
    parser = argparse.ArgumentParser(description="Visualise kernels")
    parser.add_argument("files", nargs='+', type=str, help="files with kesi kernels")

    args = parser.parse_args()



    for file in args.files:
        data = np.load(file)

        info = json.load(open(os.path.join(os.path.dirname(file), 'info.json')))

        kernel = data['KERNEL']
        ticks = np.arange(0, len(info['ch_names']))
        ticklabels = info['ch_names']

        fig, axs = pb.subplots(1, 2)
        c = axs[0].imshow(kernel)
        axs[0].set_title("{}\n kernel".format(file.split(os.sep)[-2]))
        axs[0].set_yticks(ticks)
        axs[0].set_xticks(ticks)

        axs[0].set_xticklabels(ticklabels)
        axs[0].set_yticklabels(ticklabels)
        fig.colorbar(c, ax=axs[0])

        c = axs[1].imshow(np.log(np.abs(kernel)))
        axs[1].set_title("{}\n kernel abs log".format(file.split(os.sep)[-2]))
        fig.colorbar(c, ax=axs[1])
        axs[1].set_yticks(ticks)
        axs[1].set_xticks(ticks)

        axs[1].set_xticklabels(ticklabels)
        axs[1].set_yticklabels(ticklabels)
        fig.tight_layout()
    pb.show()

if __name__ == '__main__':
    main()