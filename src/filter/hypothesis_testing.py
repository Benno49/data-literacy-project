import ast
import re
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.table import Table
from pandas.plotting import table

from filter.utils import corr_p_value, one_hot_groups_p_value, column_groups_p_value, one_hot_categories


def flatten(deep_list: list) -> list:
    return [x for sublist in deep_list for x in sublist]


df = pd.read_parquet('film_info_simple.parquet')
print(df.head())
print(df.keys())

# print(df['critics_score'].dropna())
# print(df['audience_score'].dropna())

'''
print(df['production_company'].dropna())

value_list = df['production_company'].dropna().tolist()
value_list = flatten(value_list)
value_series = pd.Series(value_list)
print('Warner:')
print(value_series[value_series.str.contains('warner')].value_counts())

print('Disney:')
print(value_series[value_series.str.contains('disney')].value_counts())

print('columbia')
print(value_series[value_series.str.contains('columbia')].value_counts())

print('metro-goldwyn-mayer')
print(value_series[value_series.str.contains('metro-goldwyn-mayer')].value_counts())

print('universal')
print(value_series[value_series.str.contains('universal')].value_counts())

print('fox')
print(value_series[value_series.str.contains('fox')].value_counts())

value_counts = value_series.value_counts()

pd.set_option('display.max_rows', None)
print(value_counts[value_counts >= 10])
'''


def extract_month(date: Optional[str]) -> Optional[int]:
    if date is not None:
        return int(date.split('-')[1])
    else:
        return None


# print(df['release_date'][df['release_date'].notnull()].transform(lambda x: int(x.split('-')[1])))
df['month'] = df['release_date'].transform(extract_month)
print(df['month'])

print(df['running_time'])
print(' '.join(df['running_time'][0]))

df['divergence_score'] = df['audience_score'] - df['critics_score']
print(df['divergence_score'])


# print(df['running_time'].transform(lambda x: ' '.join(x)))


def join_list(dlist):
    if isinstance(dlist, Iterable):
        return ' '.join(dlist)
    else:
        return dlist


print(df['running_time'].transform(join_list))

print(df['rottentomatoes_genre'].transform(lambda x: x.split('/')))
df['rottentomatoes_genre'] = df['rottentomatoes_genre'].transform(lambda x: x.replace(' ', ''))
df['rottentomatoes_genre'] = df['rottentomatoes_genre'].transform(lambda x: x.split('/'))
genre_list = df['rottentomatoes_genre'].tolist()
genre_list = flatten(genre_list)
print(pd.Series(genre_list).value_counts()[pd.Series(genre_list).value_counts() >= 100].keys())

genres = pd.Series(genre_list).value_counts()[pd.Series(genre_list).value_counts() >= 100].keys()


def list_contains(dlist, entry):
    if dlist is not None:
        return entry in dlist
    else:
        return False


for genre in genres:
    df[genre] = df['rottentomatoes_genre'].transform(lambda x: genre in x)

print(df['Drama'])
print(df.keys())

print(df)

print(df['audience_score'][df['Drama']].dropna())
genre_groups = [df['audience_score'][df[genre]].dropna() for genre in genres]

from scipy.stats import f_oneway, kruskal, pearsonr

print(f_oneway(*genre_groups))
print(kruskal(*genre_groups))

import scikit_posthocs as sp

print(sp.posthoc_ttest(genre_groups, equal_var=False))
pc = sp.posthoc_ttest(genre_groups, equal_var=False)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
# sp.sign_plot(pc, **heatmap_args)

print(type(pc))
x = np.array([[1, 1, 1],
              [1, 1, 0],
              [1, 0, 1]])
print(pc.to_numpy().tolist())
print((np.array(pc.to_numpy()) <= 0.05).astype(int).tolist())
p_005 = (np.array(pc.to_numpy()) <= 0.05).astype(int)
ax = sp.sign_plot(p_005, flat=True)

ax.set_xticklabels(['Drama', 'Comedy', 'Mystery&thriller', 'Action', 'Romance', 'Horror',
                    'Adventure', 'Crime', 'Kids&family', 'Sci-fi', 'Documentary', 'Fantasy',
                    'History', 'Biography', 'Musical', 'Western', 'War', 'Holiday',
                    'Lgbtq+', 'Animation'], fontdict=None, minor=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

ax.set_yticklabels(['Drama', 'Comedy', 'Mystery&thriller', 'Action', 'Romance', 'Horror',
                    'Adventure', 'Crime', 'Kids&family', 'Sci-fi', 'Documentary', 'Fantasy',
                    'History', 'Biography', 'Musical', 'Western', 'War', 'Holiday',
                    'Lgbtq+', 'Animation'], fontdict=None, minor=False)
plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

plt.gcf().subplots_adjust(bottom=0.22, left=0.2)
plt.show()

print('Genre:')
p_map = one_hot_groups_p_value(df, 'audience_score',
                               ['Drama', 'Comedy', 'Mystery&thriller', 'Action', 'Romance', 'Horror',
                                'Adventure', 'Crime', 'Kids&family', 'Sci-fi', 'Documentary', 'Fantasy',
                                'History', 'Biography', 'Musical', 'Western', 'War', 'Holiday',
                                'Lgbtq+', 'Animation'], 0.05, 'test.png')
print(p_map)

targets = ['audience_score', 'critics_score', 'divergence_score']
metric_params = ['year', 'rottentomatoes_length', 'budget', 'box_office']

metric_p_df = pd.DataFrame(index=targets)
for param in metric_params:
    param_p_list = []
    for target in targets:
        p = corr_p_value(df, 'audience_score', 'budget')
    metric_p_df[param] = pd.Series(param_p_list)

print(corr_p_value(df, 'audience_score', 'budget'))
print(corr_p_value(df, 'audience_score', 'box_office'))
print(corr_p_value(df, 'audience_score', 'year'))
print(corr_p_value(df, 'audience_score', 'rottentomatoes_length'))

print(corr_p_value(df, 'critics_score', 'budget'))
print(corr_p_value(df, 'critics_score', 'box_office'))
print(corr_p_value(df, 'critics_score', 'year'))
print(corr_p_value(df, 'critics_score', 'rottentomatoes_length'))

print(corr_p_value(df, 'divergence_score', 'budget'))
print(corr_p_value(df, 'divergence_score', 'box_office'))
print(corr_p_value(df, 'divergence_score', 'year'))
print(corr_p_value(df, 'divergence_score', 'rottentomatoes_length'))

column_groups_p_value(df, 'critics_score', 'month', 0.05, 'month_p.png', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def streaming_suppliers(suppliers: str):
    streaming_suppliers = []
    if suppliers is None:
        return streaming_suppliers
    for supplier, offer in ast.literal_eval(suppliers):
        if offer == 'Subscription':
            streaming_suppliers.append(supplier)
    return streaming_suppliers


def flatten_list(deeplist):
    return [entry for entrylist in deeplist for entry in entrylist]


df['suppliers'] = df['suppliers_list'].transform(lambda x: streaming_suppliers(x))

df, supplier_list = one_hot_categories(df, 'suppliers', 100)

print('-----------------------------------')

df["proportional_profit"] = df["box_office"].to_numpy() / df["budget"].to_numpy()
df["proportional_profit"] = df["proportional_profit"].transform(lambda x: 0 if np.isnan(x) or np.isinf(x) else x)

targets = ['audience_score', 'critics_score', 'divergence_score']
metric_params = ['year', 'rottentomatoes_length', 'budget', 'box_office', 'proportional_profit']

metric_p_df = pd.DataFrame(index=targets)
for param in metric_params:
    param_p_list = []
    for target in targets:
        p = corr_p_value(df, target, param)
        param_p_list.append(p)
    metric_p_df[param] = pd.Series(param_p_list, index=targets, dtype='float64')

print(metric_p_df)
df['genre'] = df['rottentomatoes_genre']
print(df['genre'])
group_params = ['genre', 'suppliers', 'month']
group_params_type = ['onehot', 'onehot', 'categoriel']
var_p_df = pd.DataFrame(index=targets)
prozent_p_df = pd.DataFrame(index=targets)
for param, param_type in zip(group_params, group_params_type):
    param_p_list = []
    param_proz_list = []
    if param_type == 'onehot':
        df, group_list = one_hot_categories(df, param, 100)
    for target in targets:
        if param_type == 'onehot':
            p, p_df = one_hot_groups_p_value(df, target, group_list, 0.05, f'{param}.png')
        elif param_type == 'categoriel':
            p, p_df = column_groups_p_value(df, target, param, 0.05, f'{param}.png')
        else:
            raise ValueError()
        print(p_df)
        print(p_df.shape[0])
        p_proz = (p_df <= 0.05).sum().sum() / (p_df.size - p_df.shape[0])
        print(p_proz)
        param_p_list.append(p)
        param_proz_list.append(p_proz)
    var_p_df[param] = pd.Series(param_p_list, index=targets, dtype='float64')
    prozent_p_df[param] = pd.Series(param_proz_list, index=targets)

print(var_p_df)
print(prozent_p_df)


def color_significant_green(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    print('val:', val)
    color = 'green' if val <= 0.05 else 'black'
    return 'color: % s' % color


# var_p_df.style.applymap(color_significant_green)


metric_p_df_str = metric_p_df.applymap('{:,.1e}'.format)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cl_outcomes = {
    'white': '#FFFFFF',
    'gray': '#AAA9AD',
    'black': '#313639',
    'purple': '#AD688E',
    'orange': '#D18F77',
    'yellow': '#E8E190',
    'ltgreen': '#CCD9C7',
    'dkgreen': '#006400',
}

faded_text_color = cl_outcomes['gray']
significant_text_color = cl_outcomes['dkgreen']

print(metric_p_df_str.values)

def plot_with_significance(df, colLabels=None, colWidths=None, save_path=None):
    df_str = df.applymap('{:,.1e}'.format)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    if colLabels is None:
        colLabels = df_str.columns
    if colWidths is None:
        colWidths = [0.3] * len(df_str.columns)
    t: Table = ax.table(cellText=df_str.values, colWidths=colWidths,
                        colLabels=colLabels,
                        loc='center', rowLabels=df_str.index, edges='TR')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    print(df.shape)
    print(t[1, 2].get_text())
    for j in range(df.values.shape[1]):
        cell: Rectangle = t[0, j]
        cell.visible_edges = 'BRL'
        for i in range(df.values.shape[0]):
            print(df.values[i, j])
            cell = t[i + 1, j]
            cell.get_text().set_horizontalalignment('center')
            if df.values[i, j] >= 0.05:
                cell.get_text().set_color(faded_text_color)
    for i in range(df.values.shape[0]):
        cell: Rectangle = t[i, df.values.shape[1]-1]
        cell.visible_edges = 'BL'
    cell: Rectangle = t[df.values.shape[0], df.values.shape[1]-1]
    cell.visible_edges = 'L'

    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

def pretty_plot(df, colLabels=None, colWidths=None, save_path=None):
    df_str = df.applymap('{:.2%}'.format)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    if colLabels is None:
        colLabels = df_str.columns
    if colWidths is None:
        colWidths = [0.3] * len(df_str.columns)
    t: Table = ax.table(cellText=df_str.values, colWidths=colWidths,
                        colLabels=colLabels,
                        loc='center', rowLabels=df_str.index, edges='TR')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    print(df.shape)
    print(t[1, 2].get_text())
    for j in range(df.values.shape[1]):
        cell: Rectangle = t[0, j]
        cell.visible_edges = 'BRL'
        for i in range(df.values.shape[0]):
            print(df.values[i, j])
            cell = t[i + 1, j]
            cell.get_text().set_horizontalalignment('center')
    for i in range(df.values.shape[0]):
        cell: Rectangle = t[i, df.values.shape[1]-1]
        cell.visible_edges = 'BL'
    cell: Rectangle = t[df.values.shape[0], df.values.shape[1]-1]
    cell.visible_edges = 'L'

    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')


plot_with_significance(var_p_df, save_path='var_p.png')
plot_with_significance(metric_p_df, colLabels=['year', 'length', 'budget', 'box_office', 'proportional_profit'], colWidths=[0.25,0.25,0.25,0.25,0.4], save_path='metric_p.png')
pretty_plot(prozent_p_df, save_path='prozent.png')
# *GENRE_VECTOR,
#     *SUPPLIER_VECTOR,
#     "rottentomatoes_length",
#     "year",
#     "month",
#     "box_office",
#     "budget",
