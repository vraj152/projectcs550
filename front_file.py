import content_based_recsys
import pandas as pd
import math

training_data = pd.read_csv(r'data/trainingData.csv')
movie_data = pd.read_csv(r'data/movieWithVector.csv')
pantomath = pd.merge(training_data, movie_data, on="movieId")

pantomath.to_csv(r'data/pantomath.csv', index=False)

def buildUserProfile(userInput):
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

    weight_list = [0.00000000000000000000000000001 if math.isnan(x) else x for x in weight_list]

    return weight_list

def take_userInput(user_Id):
    recemmendations = []
    itemProfile = r'data/itemProfile.csv'
    movies = pd.read_csv(r'data/movies.csv')
    
    weightList = buildUserProfile(user_Id)
    cosine_distances = content_based_recsys.computeSimilarity(itemProfile, weightList)
    count = 1
    
    for key, value in cosine_distances.items():
        if (count < 11):
            temp_dict = {}
            temp_dict['Index'] = count
            temp_dict['MovieName'] = str(getMovieName(movies, key))
            temp_dict['MovieID'] = key
            temp_dict['Similarity'] = value
            count = count + 1
            
            recemmendations.append(temp_dict)
        else:
            break
    return recemmendations

def getMovieName(movies, movieId):
    data = movies.loc[movies['movieId'] == movieId]
    return (data.iloc[0]['title'])
