import pandas as pd

trainingData = pd.read_csv(r'trainingData.csv')
isUser = trainingData.userId == 1
ismovie = trainingData.movieId == 1

row = trainingData[isUser & ismovie].rating.item()
print(row)
