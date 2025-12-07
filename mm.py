import pandas as pd
data=pd.read_csv('Clean_Dataset.csv')
data = data.drop(['id','flight'], axis=1)
col_name=['airline','source_city','departure_time','stops','arrival_time','destination_city','class','duration','days_left','price']
y=data['price']
X = data.drop('price', axis=1)
categorical_cols = ['airline','source_city','departure_time','stops','arrival_time','destination_city','class']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
print(X_encoded.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X_encoded,y,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
model=RandomForestRegressor(
    n_estimators=18,
    max_depth=14,
    random_state=1
)
model.fit(X_train,y_train)
ypre=model.predict(X_test)
mae=mean_absolute_error(y_test,ypre)
import numpy as np
rmse=np.sqrt(mean_squared_error(y_test,ypre))
print("MAE:", mae)
print("RMSE:", rmse)

import joblib
joblib.dump(model, 'flight_price_model.pkl')
joblib.dump(X_encoded.columns, 'columns.pkl')
model = joblib.load('flight_price_model.pkl')
columns = joblib.load('columns.pkl')

feature_list = ['airline','source_city','departure_time','stops','arrival_time',
                'destination_city','class','duration','days_left']

customer = {}
print("Enter the details of the flight:")

for feature in feature_list:
    value = input(f"{feature}: ")
    if feature in ['duration','days_left']:
        value = float(value)
    customer[feature] = value
df = pd.DataFrame([customer])
df_encoded = pd.get_dummies(df)

for col in columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[columns]
pred_price = model.predict(df_encoded)[0]
print(f"\nPredicted Flight Price: {pred_price:.2f}")
