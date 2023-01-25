import re
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from filter.utils import corr_p_value, one_hot_groups_p_value, column_groups_p_value


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

from scipy.stats import f_oneway, kruskal

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

print(corr_p_value(df, 'audience_score', 'budget'))
print(corr_p_value(df, 'audience_score', 'box_office'))
print(corr_p_value(df, 'audience_score', 'year'))
print(corr_p_value(df, 'audience_score', 'rottentomatoes_length'))

print(corr_p_value(df, 'critics_score', 'budget'))
print(corr_p_value(df, 'critics_score', 'box_office'))
print(corr_p_value(df, 'critics_score', 'year'))
print(corr_p_value(df, 'audience_score', 'rottentomatoes_length'))

column_groups_p_value(df, 'critics_score', 'month', 0.05, 'month_p.png', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
