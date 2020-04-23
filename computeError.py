import pandas as pd
import content_based_recsys
import math

# Function to create predictedRatingTestData file which contains user, movieId, predictedRating and givenRating
def createPredictedFile(user, movieId, predictValue, givenRating, predictedRatingTestData):
    row = [{'user': user, 'movieId': movieId, 'predictedRating': predictValue, 'actualRating': givenRating}]
    predictedRatingTestData = predictedRatingTestData.append(row, ignore_index = True, sort = False)
    return predictedRatingTestData

# Function to predict the movie ratings by using knn algorithm
def predict(cosineDistances, user):
    count = 1
    pSum = 0
    denom = []
    for key, value in cosineDistances.items():
        isUser = trainingData.userId == user
        ismovie = trainingData.movieId == key
        if count < 20:
            userRating = trainingData[isUser & ismovie].rating.item()
            pSum = pSum + (value*userRating)
            denom.append(value)
            count = count + 1

    if sum(denom) != 0:
        predictValue = pSum / (sum(denom))
    else:
        v = 0.0000000001
        predictValue = pSum / v

    return predictValue


# Function to calculate rmse
def rmse(predictedList, predictedRatingTestData):
    n = 0.0
    errorSum = 0.0

    for value in predictedList:
        givenRating = testData.loc[(testData['movieId'] == value[1][1]) & (testData['userId'] == value[0])].rating.item()
        v = value[1][2]
        errorDiff = (v - givenRating) ** 2
        errorSum = errorSum + errorDiff
        n = n+1

        predictedRatingTestData = createPredictedFile(value[0], value[1][1], v, givenRating, predictedRatingTestData)

    answer = math.sqrt(errorSum/n)
    print("RMSE = ", answer)
    return predictedRatingTestData


trainingData = pd.read_csv(r'data/trainingData.csv')

testData = pd.read_csv(r'data/testingData.csv')
allUsers = testData['userId'].unique()
itemProfile = pd.read_csv(r'data/itemProfile.csv')
fileCol = itemProfile.columns
predictedList = []

cols = ["user", "movieId", "predictedRating", "actualRating"]
predictedRatingTestData = pd.DataFrame(columns = cols)

for user in allUsers:
    userFile = pd.DataFrame(columns = fileCol)
    userTrainFile = pd.DataFrame(columns = fileCol)

    #Create file for test data
    isUser = testData.userId == user
    movies = testData[isUser].movieId

    for index, eachMovie in movies.iteritems():
        isMovie = itemProfile.movieId == eachMovie
        row = itemProfile[isMovie]
        userFile = userFile.append(row, ignore_index=True, sort=False)

    userFile.to_csv(r'data/userFile.csv')

    #Create movies rated by users in trained data
    isUser = trainingData.userId == user
    movies = trainingData[isUser].movieId

    for index, eachMovie in movies.iteritems():
        isMovie = itemProfile.movieId == eachMovie
        row = itemProfile[isMovie]
        userTrainFile = userTrainFile.append(row, ignore_index=True, sort=False)

    userTrainFile.to_csv(r'data/userTrainFile.csv', index=False)

    userFile = pd.read_csv(r'data/userFile.csv')
    address = r'data/userTrainFile.csv'
    for index, row in userFile.iterrows():
        cosineDistances = content_based_recsys.computeSimilarity(address, row[2:22].values.tolist())
        predictValue = predict(cosineDistances, user)
        predictedList.append((user, (index+1, row.movieId.item(), predictValue)))  # Change for all users

predictedRatingTestData = rmse(predictedList, predictedRatingTestData)
predictedRatingTestData.to_csv(r'data/predictedRatingTestData.csv', index=False)