import pandas as pd

pd.set_option('display.max_columns', None)

part_nr = 3
df = pd.read_csv(f"film_info_part_{part_nr}.csv", dtype=str)
print(df.head())
print(df.keys())
print(df.shape)
print('duplicated:')
print(df[df.duplicated(['title', 'year'])])

df_wiki = pd.read_csv(f"film_wiki_part_{part_nr}.csv", dtype=str)
print('df_wiki:')
print(df_wiki.keys())
print(df_wiki[df_wiki.duplicated(['title', 'year'], keep=False)])
df_wiki.drop_duplicates(inplace=True)
print(df_wiki[df_wiki.duplicated(['title', 'year'], keep=False)])

df.set_index(['title', 'year'], drop=True, inplace=True)
df.drop(columns=['infobox'], inplace=True)

print(df.shape)
print(df.keys())
df_wiki.set_index(['title', 'year'], drop=True, inplace=True)
df = df.join(df_wiki, how='inner')
print(df.head())
print(df.keys())
print(df.shape)

df.reset_index(drop=False, inplace=True)
df.to_csv(f"film_info_part_{part_nr}.csv")
