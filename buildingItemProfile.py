#%%
import pandas as pd

#%%
movie_metadata = pd.read_csv(r'data/movies.csv')
unique_comb = movie_metadata.genres.unique()
unique_genre = ["movieId"]

for each_row in unique_comb:
    parts = each_row.split('|')
    for each_part in parts:
        if each_part not in unique_genre:
            unique_genre.append(each_part)

#%%
column_names_item_profile = unique_genre
item_profile = pd.DataFrame(columns = column_names_item_profile)

for index, row in movie_metadata.iterrows():
    parts = row.genres.split('|')
    temp = dict.fromkeys(unique_genre,0)
    for each_part in parts:
        temp[each_part] = 1
    item_profile = item_profile.append(temp, ignore_index= True, sort=False)

for index, row in movie_metadata.iterrows():
    movieId = row.movieId
    item_profile.xs(index)['movieId']=movieId
    
#%%
for index, row in movie_metadata.iterrows():
    movieId = row.movieId
    item_profile.xs(index)['movieId']=movieId

item_profile.to_csv(r'data/itemProfile.csv', index=False)

#%%
result = pd.merge(movie_metadata, item_profile, on = "movieId")
result.to_csv(r'data/movieWithVector.csv', index=False)