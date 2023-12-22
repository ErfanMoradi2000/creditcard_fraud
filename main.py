import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('data/creditcard.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

model = KMeans(n_clusters=2, n_init=10)
model.fit(X=X)
y_predicted = model.predict(X)

n = (y == y_predicted).sum()

