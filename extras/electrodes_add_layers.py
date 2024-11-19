import argparse

import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser(description="converts .npz sampled solution file to nifti")
    parser.add_argument("files", nargs='+')
    args = parser.parse_args()
    for file in args.files:
        df = pd.read_csv(file)
        df_orig = df.copy()
        positions = df[['x', 'y', 'z']].values
        positions_norm = np.sqrt(np.sum(positions**2, axis=1))[:, None]
        depths = [65.0, 35.0]
        for nr, depth in enumerate(depths):
            name = "_L{}".format(nr+1)
            names = list(df_orig['label'])
            new_names = [i+name for i in names]
            new_pos = positions / positions_norm * depth
            new_df = df_orig.copy()
            new_df['label'] = new_names
            new_df['x'] = new_pos[:, 0]
            new_df['y'] = new_pos[:, 1]
            new_df['z'] = new_pos[:, 2]
            df = pd.concat([df, new_df])
        df.to_csv(file, index=None)


if __name__ == '__main__':
    main()