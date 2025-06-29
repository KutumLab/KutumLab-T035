from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from tqdm import tqdm


def perform_ttest(
    xx, yy, equal_var: bool = False, class0: int = 0, class1: int = 1
) -> Tuple[float, float, float]:
    """
    Perform ttest and return pval and group mean

    Args:
        x: array like
        y: array like (binary category)
        equal_var: The variance of x and y is equal (False for Welch Two Sample t-test)
    Returns:
        p-val, group 0 mean , grpup 1 mean
    """

    try:
        assert len(xx) == len(yy)
    except AssertionError as e:
        print(f"Assertion Error: {e} Unequal lenghth of xx and yy.")

    result = ttest_ind(xx, yy, equal_var=equal_var)
    pval = result.pvalue

    # Group 0 and Group 1 based on yy
    group0 = xx[yy == class0]
    group1 = xx[yy == class1]

    # Calculate means
    mean_group0 = np.mean(group0)
    mean_group1 = np.mean(group1)

    # print("Mean in group 0:", mean_group0)
    # print("Mean in group 1:", mean_group1)

    return pval, mean_group0, mean_group1


def calculate_var(
    mol_desc: pd.DataFrame, yy
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vars = []
    pvals = []
    means_group0 = []
    means_group1 = []
    for col in tqdm(mol_desc.columns, total=len(mol_desc.columns)):
        vals = mol_desc[col].values
        var = np.var(vals)
        pval, m0, m1 = perform_ttest(vals, yy)

        vars.append(var)
        pvals.append(pval)
        means_group0.append(m0)
        means_group1.append(m1)

    vars = np.array(vars)
    pvals = np.array(vars)
    means_group0 = np.array(means_group0)
    means_group1 = np.array(means_group1)

    assert len(vars) == len(pvals)
    assert len(vars) == len(means_group0)
    assert len(vars) == len(means_group1)

    return vars, pvals, means_group0, means_group1


def filter_variance(
    df: pd.DataFrame, variance: np.ndarray, threshold: float = 0.1, top_n=None
) -> np.ndarray:
    """
    Threshold the 'variance'. Returns the index of features in decending order of 'variance'.
    """

    features = df.columns[variance > threshold]
    threshold_variance = pd.Series(variance[variance > threshold])
    threshold_variance = threshold_variance.sort_values(ascending=False)
    features = features[threshold_variance.index].values
    if top_n is not None:
        features = features[:top_n]
    return features


def filter_diff_abs(
    df: pd.DataFrame,
    means_group0: np.ndarray,
    means_group1: np.ndarray,
    threshold: float = 0.5,
    do_sort:bool=True
) -> np.ndarray:
    """
    Threshold the 'group mean diff'. Returns the index of features in decending order of 'group mean diff'.
    """
    abs_diff = np.abs(means_group0 - means_group1)
    features = df.columns[abs_diff > threshold]
    threshold_abs_diff = pd.Series(abs_diff[abs_diff > threshold])
    if do_sort:
        threshold_abs_diff = threshold_abs_diff.sort_values(ascending=False)
        features = features[threshold_abs_diff.index].values
    else:
        features = features.values

    return features


def plot_scatter_vars(
    means_group0,
    means_group1,
    vars,
    title: str,
    save_path: str,
    threshold: float = 0.1,
    do_save: bool = False,
):
    plt.figure(figsize=(3.5, 3.5))
    N = len(vars)
    x = vars
    y = np.array(means_group0 - means_group1)

    sns.scatterplot(
        x=x,
        y=y,
        legend="brief",
    )

    plt.xlabel("variance")
    plt.ylabel("group mean diff")
    plt.title(f"Filtered ({threshold}, N={N}): {title}")

    # min_ = min(min(x), min(y)) - 0.5
    # max_ = max(max(x), max(y)) + 1

    # plt.xlim(min_, max_)
    # plt.ylim(min_, max_)

    plt.tight_layout()

    if do_save:
        plt.savefig(save_path)

    plt.show()
