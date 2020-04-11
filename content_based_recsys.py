import pandas as pd
from scipy import spatial
import math

def build_user_profile(userInput):
    item_profile = pd.read_csv(r'data/itemProfile.csv')
    training_data = pd.read_csv(r'data/trainingData.csv')
    movie_data = pd.read_csv(r'data/movieWithVector.csv')
    pantomath = pd.merge(training_data, movie_data, on="movieId")
    
    pantomath.to_csv(r'data/pantomath.csv', index=False)
    
    rated_movies = pantomath.loc[pantomath['userId'] == int(userInput)]
    
    user_mean_rating = round(rated_movies["rating"].mean(), 2)
    updated_rating = []
    
    for each_rate in rated_movies["rating"]:
        updated_rating.append(each_rate - user_mean_rating)
    
    rated_movies.insert(loc=4, column='updatedRating', value=updated_rating)
    
    features = rated_movies.columns[7:27]
    weight_dict = {}
    weight_list = []
    
    for each_feature in features:
        set_of_feature = rated_movies.loc[rated_movies[each_feature] == 1]
        length_of_feature = len(set_of_feature)
        sum_of_feature = set_of_feature["updatedRating"].sum()
        weight_of_feature = sum_of_feature / length_of_feature
        weight_dict[each_feature] = weight_of_feature
        weight_list.append(weight_of_feature)
    
    each_item_vector = item_profile.iloc[:,1:21].values
    each_item_id = item_profile.iloc[:,0].values
    weight_list = [0.00000000000000000000000000001 if math.isnan(x) else x for x in weight_list]    
    cosine_distances = {}
    i = 0
    
    while (i != len(each_item_vector)):
        cosine_distance = spatial.distance.cosine(weight_list,each_item_vector[i])
        cosine_distances[each_item_id[i]] = (1-cosine_distance)
        i = i + 1
    cosine_distances = {k: v for k, v in sorted(cosine_distances.items(), key=lambda item: item[1], reverse=True)}
    return weight_dict, cosine_distances, weight_list