import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# Data Import
df = pd.read_csv('Fruit_Prices.csv')
#print(df)
#print(df.dtypes)
#print(df.describe)
#print(df.isnull().sum())


# Data Transformation
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


# EDA
#plt.figure(figsize=(8, 8))
#sns.heatmap(df.corr(), annot=True, cbar=False, cmap='crest')
#plt.show()


# Data Splitting
features = df.drop(['RetailPrice'], axis=1)
target = df['RetailPrice']

X_train, X_val, Y_train, Y_val = train_test_split(features,target, test_size=0.2, random_state=12)
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


# Model Training
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

models = [LinearRegression(), RandomForestRegressor()]

for m in models:
    m.fit(X_val, Y_val)
    Y_pred = m.predict(X_val)
    print(f'Acuracy of {m} : {1 - (mae(Y_val, Y_pred))}')

