"""
This module contains the main functionality of the normal distribution fitting approach.

Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht


"""

# FIT NORMAL DISTRIBUTION TO EVALUATED DISTANCES TO EVALUATE SIGNIFICANT DISTANCES BY P-VALUES
# Distances from Pf-antigen influence evaluation is fit to a normal distribution to obtain those with p-value <= 0.05

# This feature evaluation method can be only applied on the output files from
# Feature_Evaluation_from_multitask_SVM_approach_per_timepoint.py method.

import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def extraction_90percentage_of_Data(data):
    """ Extract pre-defined percentage of evaluated distances

            Features are sorted by their distance value and splitted into two sets: one set containing 95% of the
            features and one set containing 5% of the features.


            Args: data (matrix): matrix of evaluated distances per feature, d x m (d = number of distances, m = number
                                 of features)

            Returns: data (matrix): matrix of 95% of evaluated distances per feature
                     excl_ten_per_largest_distances (matrix): matrix of the 5% of features
    """
    plt.hist(abs(data.loc["|d|"].values), bins=50, density=True, alpha=0.6, color='b')
    plt.xticks(rotation=90)
    title = "Distance distribution of whole data"
    plt.title(title)
    plt.show()
    # calc 5% of number of distances
    five_per = (len(data.columns)/100)*5  # for simulated data *1.5
    five_per = math.ceil(five_per)
    data = data.T
    # define 5% of the n-largest distance values
    excl_five_per_largest_distances = pd.DataFrame(data.nlargest(five_per, ["sort"]))
    # print("Length of 5% of the n largest ESPY values is:", len(excl_five_per_largest_distances.columns))
    data = data[~data.isin(excl_five_per_largest_distances)].dropna()

    return data, excl_five_per_largest_distances


def map_on_normal_distribution(data):
    """ Map evaluated distances of features to a normal distribution

            95% of the feature distances are mapped to a normal distribution.

            Args: data (matrix): matrix of 95% of evaluated distances per feature

            Returns: right_tail (float): position of 5% tail on positive x axis (right tail of distribution)
                     left_tail (float): position of 5% tail on negative x axis (left tail of distribution)
                     mu (float): mean of distribution
                     std (float): standard deviation of distribution
    """
    # plot original distribution of 95% of the data
    # plt.hist(data["|d|"], bins=50, density=True, alpha=0.6, color='b')
    # plt.xticks(rotation = 90)
    # title_a = "Distribution of 95% of the distances"
    # plt.title(title_a)
    # plt.show()

    # Fit a normal distribution to the data:
    a, loc, scale = gamma.fit(abs(data["|d|"]))
    print("a paramter: " + str(a) + 'loc: ' + str(loc) + "scale: " + str(scale))
    # Plot the PDF (probability density function) - normal fitted data.
    plt.hist(abs(data["|d|"]), bins=50, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = gamma.pdf(x, a, loc, scale)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xticks(rotation = 90)
    #title_b = "Fitted Data on normal distribution with m = %.7f,  std = %.7f" % (mu, std)
    #plt.title(title_b)
    plt.show()
    # Return x-marks for 5% tail on both sides
    # right_tail = norm.ppf(.95, mu, std)
    # left_tail = norm.ppf(.05, mu, std)
    # print("x-marks of tails:", "\n", "right tail = ", right_tail, "\n", "left_tail = ", left_tail)

    return a, loc, scale


def calc_p_values(data, a, loc, scale):
    """ Calculate p-values for each feature

        calculate p-value for each feature based on normal distribution fitting

            Args: data (matrix): matrix of the 5% of features
                  right_tail (float): position of 5% tail on positive x axis (right tail of distribution)
                  left_tail (float): position of 5% tail on negative x axis (left tail of distribution)
                  m (float): mean of distribution
                  std (float): standard deviation of distribution

            Returns: results_table (matrix): table of p-values per feature and the significance level (p-value < 0.05)
    """
    result_table = pd.DataFrame(data["|d|"])

    # calc p-values for Pf-antigen distance
    cum_values = []
    for d in data["|d|"]:
        cum_value = gamma.cdf(d, a, loc, scale)
        cum_values.append(cum_value)

    # add p-values for Pf-antigen to result table
    result_table["cum_values"] = cum_values

    # add significance status for p-value
    status = []
    for p in result_table["cum_values"]:
        if p > 0.95:  # < 0.05: or p > 0.95:  # (p < 0.05 or p > 0.95):
            status.append("significant")
        else:
            status.append("non_significant")
    # add status to result table
    result_table["significant level"] = status
    print("sum significant features: ")
    print(sum(result_table.loc[:, "significant level"].isin(["significant"])))
    return result_table


def filter_significant_Pfantigens(data, significant_Pfantigen_table):
    """ Filter significant features

            Extract significant features based on p-value < 0.05.

            Args: data (matrix): matrix of evaluated distances per feature, d x m (d = number of distances, m = number
                                 of features)
                  significant_Pfantigen_table (matrix): matrix of features with p-value < 0.05

            Returns: data (matrix): table of evaluated significant features and their evaluated distances d x m (d =
                                    number of distances, m = number of features, with p-value < 0.05).
    """
    data = data.T
    array = significant_Pfantigen_table.loc[significant_Pfantigen_table.loc[:, "significant level"].isin(["significant"])]
    #print(array)
    data = data.loc[data.index.isin(array.index)]
    #print(data)
    print("Dimension of final significant Pfantigens", data.shape)
    data = data.T
    data.loc["|d|"] = abs(data.loc["|d|"].values)
    return data


def make_plot(data, name, outputdir):
    plt.figure(figsize=(20, 10))
    labels = data.columns

    ax = plt.subplot(111)
    w = 0.3
    opacity = 0.6

    index = np.arange(len(labels))
    ax.bar(index, abs(data.loc["|d|"].values), width=w, color="darkblue", align="center", alpha=opacity)
    ax.xaxis_date()

    plt.xlabel('number of features', fontsize=20)
    plt.ylabel('ESPY value', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=90)

    plt.savefig(os.path.join(outputdir, name + ".png"), dpi=600)
    plt.savefig(os.path.join(outputdir, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def main_function(data, outputdir, output_name):
    """ MAIN MODUL of the normal distribution fitting approach.

                This main modul fits the evaluated distances of each feature to a normal distribution to determine
                significant features by their respective p-value < 0.05.


                Args: data (matrix): results of Parser_feature_selection.py, a matrix of d x m (d = number of
                                     distance values, m = number of features)
                      outputdir (path): path where the results will be saved
                      output_name (str): name of output picture


                Returns: significant_features (list): list of significant features
                    """

    data_90per_extraction, data_10per_extraction = extraction_90percentage_of_Data(data)
    a, loc, scale = map_on_normal_distribution(data_90per_extraction)
    result_table = calc_p_values(data_10per_extraction, a, loc, scale)
    #result_table.to_csv(os.path.join(outputdir, "Result_p_values.tsv"), sep='\t', na_rep='nan')
    print(result_table)

    significant_features = filter_significant_Pfantigens(data, result_table)
    make_plot(significant_features.iloc[:, :], output_name, outputdir)
    print("Number of evaluated significant features: " + str(len(significant_features.columns)))

    return significant_features
