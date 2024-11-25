import argparse
import json
import os.path
from email.policy import default

import numpy as np
import pylab as pb

def main():
    parser = argparse.ArgumentParser(description="Visualise kernels")
    parser.add_argument("files", nargs='+', type=str, help="files with kesi kernels")
    parser.add_argument("-c", '--channel', type=str, help="Cross section channel", default=None)


    args = parser.parse_args()

    if args.channel:
        fig = pb.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Cross sections of the matrices at channel {}".format(args.channel))


    for file in args.files:
        data = np.load(file)

        info = json.load(open(os.path.join(os.path.dirname(file), 'info.json')))

        kernel = data['KERNEL']
        ticks = np.arange(0, len(info['ch_names']))
        ticklabels = info['ch_names']
        print(ticklabels)

        fig, axs = pb.subplots(1, 2)
        c = axs[0].imshow(kernel)
        average = np.average(kernel)
        std = np.std(kernel)
        axs[0].set_title("{}\n kernel\naverage: {}\n std: {}".format(file.split(os.sep)[-2], average, std))
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

        if args.channel:
            ch_id = info['ch_names'].index(args.channel)
            ch_data = kernel[ch_id]
            ax.plot(ch_data, label=file.split(os.sep)[-2])
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)

    if args.channel:
        ax.legend()


    pb.show()

if __name__ == '__main__':
    main()