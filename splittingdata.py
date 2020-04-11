#%%
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
movie_data = pd.read_csv(r'data/ratings.csv')

#%%
column_names_X = ["userId","movieId","timestamp"]

X_train= pd.DataFrame(columns = column_names_X)
X_test= pd.DataFrame(columns = column_names_X)
y_train = pd.Series(name="rating")
y_test = pd.Series(name="rating")

totalId = movie_data.userId.unique()

for i in totalId:
    temp_data = movie_data.loc[movie_data['userId'] == i]

    y = temp_data.rating
    X = temp_data.drop('rating', axis=1)

    X_tr, X_te, y_tr, y_te = train_test_split(X , y, test_size=0.2)

    X_train = pd.concat([X_train,X_tr])
    X_test  = pd.concat([X_test,X_te])

    y_train = pd.concat([y_train,y_tr])
    y_test  = pd.concat([y_test,y_te])
    
X_train['rating'] = y_train
X_train.to_csv(r'data/trainingData.csv',index=False)

X_test['rating'] = y_test
X_test.to_csv(r'data/testingData.csv',index=False)