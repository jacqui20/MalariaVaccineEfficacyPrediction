# Copyright (c) 2022 Jacqueline Wistuba-Hamprecht and Bernhard Reuter.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# Jacqueline Wistuba-Hamprecht and Bernhard Reuter (2022)
# https://github.com/jacqui20/MalariaVaccineEfficacyPrediction
# ------------------------------------------------------------------------------------------------
# This is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------------------------

"""
Function to group strongly covarying features together.

Will save the results to .csv files

@Author: Bernhard Reuter

"""

import nestedcv as ncv
import pandas as pd
import numpy as np
import scipy
import warnings
import pathlib
import os
import argparse
import matplotlib.pyplot as plt


def main(
    data_dir,
    identifier,
    out_dir,
    timepoint,
    correlation_threshold,
    correlation_method,
):

    # some input sanity checks:
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold is not in [0, 1]")
    if correlation_method not in ['spearman', 'pearson']:
        raise ValueError("correlation_method is not in {'spearman', 'pearson'}")

    print(f"Grouping features based on {correlation_method} correlation "
          f"with threshold {correlation_threshold}:\n")
    fn = os.path.join(data_dir, f'{identifier}_data_{timepoint}.csv')
    data = pd.read_csv(fn, sep=',', index_col=0)
    print(f'Shape of dataframe loaded from {fn}: {data.shape}\n')

    # check, if there are NaN DF entries:
    if data.isna().sum().sum() != 0:
        raise ValueError(f"{fn} contains NaN entries.")

    intensities = data.drop(columns=['group', 'Protection', 'Dose', 'TimePointOrder'])
    print(f'Shape of intensities dataframe: {intensities.shape}')
    print(f'# of variants: {intensities.shape[1]}\n')

    # check if there are constant features (and drop them):
    const_features = intensities.columns[intensities.std(axis=0.0) == 0].to_list()
    if const_features:
        warnings.warn(f"{fn} contains constant features.")
        intensities.drop(columns=const_features, inplace=True)
        print(f"{fn} contains constant features:")
        for i in const_features:
            print(i)
        print(f'Shape of dataframe after dropping constant features: {intensities.shape}\n')

    # calculate either pearson or spearman correlation dataframe
    correlation = intensities.corr(method=correlation_method)
    print(f"correlation matrix shape: {correlation.shape}")
    print(
        f"correlation matrix memory usage (kB): {correlation.memory_usage(deep=True).sum()/1000}\n"
    )
    if not correlation.index.equals(correlation.columns):
        raise ValueError("correlation.index != correlation.columns")

    # # save correlation dataframe as numpy matrix
    # fn = os.path.join(
    #     out_dir,
    #     (f'{identifier}_data_{timepoint}_{correlation_method}_'
    #      f'correlation_matrix_indices_threshold_{correlation_threshold}.csv'),
    # )
    # np.savetxt(fn, correlation.index.to_list(), delimiter=',', fmt='%s')
    # fn = os.path.join(
    #     out_dir,
    #     (f'{identifier}_data_{timepoint}_{correlation_method}_'
    #      f'correlation_matrix_threshold_{correlation_threshold}.npy'),
    # )
    # np.save(fn, correlation.to_numpy(), allow_pickle=False)

    # check for variants with NaN correlation:
    if correlation.isna().sum().sum() != 0:
        raise ValueError(
            f"{fn}: There are NaN correlation matrix entries not belonging to constant features."
        )

    # construct a dict of correlated features
    if len(correlation.index) != len(set(correlation.index)):
        raise ValueError("len(correlation.index) != len(set(correlation.index))")
    all_features = set(correlation.index)
    groups = dict()
    for i in correlation.index.to_list():
        group = set(
            correlation.loc[i, ~correlation.loc[i, :].between(
                -correlation_threshold, correlation_threshold, inclusive='neither'
            )].index
        )
        groups[i] = sorted(group & all_features)
        all_features = all_features - group
    groups_values_set = set()
    for i in groups.values():
        groups_values_set.update(i)
    if groups_values_set != set(correlation.index):
        raise ValueError("groups_values_set != set(correlation.index)")

    # save dict as json:
    fn = (f'{identifier}_data_{timepoint}_{correlation_method}_correlation_'
          f'grouped_features_threshold_{correlation_threshold}')
    ncv.save_json(groups, out_dir, fn, timestamp=False)

    # keep only the key(-variant)s
    keep = []
    count = []
    for key in groups.keys():
        if len(groups[key]) > 0:
            keep.append(key)
            count.append(len(groups[key]))
        else:
            raise ValueError(f"The {key} group is empty.")
    count, keep_sorted = zip(*sorted(zip(count, keep), reverse=True, key=lambda x: x[0]))
    hist = pd.DataFrame(count, index=keep_sorted, columns=['# correlated features'])
    fn = os.path.join(
        out_dir,
        (f'{identifier}_data_{timepoint}_{correlation_method}'
         f'_correlated_group_sizes_threshold_{correlation_threshold}.csv'),
    )
    hist.to_csv(fn, sep=',')
    fn = os.path.join(
        out_dir,
        (f'{identifier}_data_{timepoint}_{correlation_method}'
         f'_correlated_group_sizes_threshold_{correlation_threshold}.pdf'),
    )
    hist_max = np.amax(hist.to_numpy())
    fig, ax = plt.subplots()
    ax.hist(hist.to_numpy(), bins=hist_max, log=True, rwidth=1.0)
    ax.set_xlim(0, hist_max + 1)
    plt.title(f"Histogram of group sizes S (S_max = #bins = {hist_max})")
    plt.savefig(fn, format='pdf')
    plt.close()
    print(f'# of (key-)variants to keep: {len(keep)}\n')
    intensities = intensities.loc[:, keep]
    metadata = data[['group', 'Protection', 'Dose', 'TimePointOrder']]
    if intensities.index.to_list() != metadata.index.to_list():
        raise ValueError("intensities.index != data.index")
    if set(intensities.columns.to_list()) & set(metadata.columns.to_list()):
        raise ValueError(
            f"Columns overlap when joining intensities dataframe with metadata for {identifier}"
            f" proteome data at time {timepoint} for threshold {correlation_threshold}."
        )
    data = metadata.join(intensities, how='left', sort=False)
    if set(data.columns.to_list()) != \
            set(intensities.columns.to_list()) | set(metadata.columns.to_list()):
        raise ValueError(
            "set(data.columns.to_list) != set(intensities.columns.to_list()) "
            "| set(metadata.columns.to_list())"
        )

    # save reduced DF:
    fn = os.path.join(
        data_dir,
        (f'{identifier}_data_{timepoint}_{correlation_method}'
         f'_decorrelated_threshold_{correlation_threshold}.csv'),
    )
    data.to_csv(fn, sep=',')

    print(f"Shape of kept {identifier} proteome data at time {timepoint}: {data.shape}")
    print('')


if __name__ == "__main__":
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    parser = argparse.ArgumentParser(
        description=('Function to group strongly covarying features together.')
    )
    parser.add_argument(
        '--data-dir', dest='data_dir', metavar='FILE', required=True,
        help='Path to the timepoint-wise proteome data files.'
    )
    parser.add_argument(
        '--identifier', dest='identifier', required=True,
        help=('Prefix to identify the proteome dataset.')
    )
    parser.add_argument(
        '--out-dir', dest='out_dir', metavar='DIR', required=True,
        help='Path to the directory to which the output shall be written.'
    )
    parser.add_argument(
        '--timepoint', dest='timepoint', required=True, type=str,
        help='Time point for which the analysis shall be performed.'
    )
    parser.add_argument(
        '--correlation_threshold', dest='correlation_threshold', required=True, type=float,
        help=(
            'The correlation coefficient threshold t determining if two features are '
            'grouped together: if |correlation| >= t they are grouped.'
        )
    )
    parser.add_argument(
        '--correlation_method', dest='correlation_method', required=True, type=str,
        help=(
            "The method used to calculate the correlation. "
            "Use either 'spearman' (Spearman rank correlation) or 'pearson' (Pearson correlation)."
        )
    )
    args = parser.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    main(
        args.data_dir,
        args.identifier,
        args.out_dir,
        args.timepoint,
        args.correlation_threshold,
        args.correlation_method,
    )
