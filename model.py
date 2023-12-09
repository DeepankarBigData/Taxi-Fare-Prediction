import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("dj.csv")

X = df.drop(['key', 'pickup_datetime', 'pickup_weekday', 'fare_amount', 'pickup_day','pickup_hour','pickup_date','pickup_month'], axis=1)

y = df['fare_amount']

X_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12345)

model = RandomForestRegressor(n_estimators=100, max_depth=11, random_state=42)

mod = model.fit(X_train, y_train)

pickle.dump(mod, open("model.pkl", "wb"))

