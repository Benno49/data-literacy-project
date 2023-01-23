from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import SubplotBase
from scipy.stats import f_oneway, linregress
import scikit_posthocs as sp


def flatten(deep_list: list) -> list:
    return [x for sublist in deep_list for x in sublist]


def list_contains(dlist, entry):
    if dlist is not None:
        return entry in dlist
    else:
        return False


def one_hot_categories(df: pd.DataFrame, column: str, min_occurences: int,
                       alias_dict: Optional[dict] = None) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    entry_list = flatten(df[column].tolist())

    value_list = pd.Series(entry_list).value_counts()[pd.Series(entry_list).value_counts() >= min_occurences].keys()

    for value in value_list:
        df[value] = df['column'].transform(lambda x: list_contains(x, value))

    return df, value_list


def parse_genre(genre_entry: str) -> list[str]:
    genre_entry = genre_entry.replace(' ', '')
    genre_list = genre_entry.split('/')
    return genre_list


def cal_and_plot_p_values(x, groups_list: list, p_value: float, save_path: str):
    print(f_oneway(*x))

    pc = sp.posthoc_ttest(x, equal_var=False)

    p_005 = (np.array(pc.to_numpy()) <= p_value).astype(int)
    ax: SubplotBase = sp.sign_plot(p_005, flat=True)

    ax.set_xticklabels(groups_list, fontdict=None, minor=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_yticklabels(groups_list, fontdict=None, minor=False)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    plt.gcf().subplots_adjust(bottom=0.22, left=0.2)
    plt.savefig(save_path)


def one_hot_groups_p_value(df: pd.DataFrame, score_column: str, groups_list: list, p_value: float, save_path: str):
    x = [df[score_column][df[bool_column]].dropna() for bool_column in groups_list]

    cal_and_plot_p_values(x, groups_list, p_value, save_path)


def column_groups_p_value(df: pd.DataFrame, score_column: str, groups_column: str, p_value: float, save_path: str,
                          groups_list: Optional[list] = None):
    if groups_list is None:
        groups_list = df[groups_column].value_counts().keys()
    x = [df[score_column][df[groups_column] == group].dropna() for group in groups_list]

    cal_and_plot_p_values(x, groups_list, p_value, save_path)


def corr_p_value(df: pd.DataFrame, score_column: str, value_column: str):
    df = df[[score_column, value_column]].dropna()
    print(linregress(df[score_column], df[value_column]))
