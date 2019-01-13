import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

games  = pd.read_csv("games.csv")

plt.hist(games["average_rating"])

print(games[games["average_rating"]==0].iloc[0])

print(games[games["average_rating"] > 0].iloc[0])

games=games[games["users_rated"] > 0]

games = games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show()
print(games.columns)

#correlationm matrix

cormat = games.corr()

fig = plt.figure(figsize = (12,9))

sns.heatmap(cormat, vmax= .8, square = True)
plt.show()

columns= games.columns.tolist()

columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

target = "average_rating"

from sklearn.model_selection import train_test_split

train = games.sample(frac=0.8, random_state = 1)

test=games.loc[~games.index.isin(train.index)]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr=LinearRegression()

print(lr.fit(train[columns],train[target]))
        
predict = lr.predict(test[columns])

print(mean_squared_error(predict,test[target]))

from sklearn.ensemble import RandomForestRegressor

rfr= RandomForestRegressor(n_estimators=100, min_samples_leaf=10,random_state=1)

print(rfr.fit(train[columns],train[target]))
predictions = rfr.predict(test[columns])

print(mean_squared_error(predictions,test[target]))

print(test[columns].iloc[0])
rating_lr=lr.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_rfr=rfr.predict(test[columns].iloc[0].values.reshape(1,-1))

print(rating_lr)
print(rating_rfr)

print(test[target].iloc[0])
