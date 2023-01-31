import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('housing.csv')

print("observing data", data.info())
print('observing shape of data', data.shape)
print("observing null values", data.isnull().sum())
print("observing null values", data.isnull().sum().sum())

# total entries 20640 out of which null values 207 under total_bedrooms
correlation = data.corr(numeric_only=True)
sns.heatmap(correlation, annot=True)
plt.show()

# correlation between total_bedrooms and total rooms have .93 correlation - filling null values with total_rooms

data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
print("observing null values", data.isnull().sum().sum())

# replacing Ocean_proximity with integers

data = data.replace({'ocean_proximity': {'<1H OCEAN': 0, 'INLAND': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}})
print(data['ocean_proximity'].value_counts())

# ********* unwanted trial converting latitude and longitude to area wise split

# print(data['longitude'].value_counts())
# X = data['longitude']
# print(data['longitude'].value_counts())
# X = round(data['longitude'])
# Y = round(data['latitude'])
# print(X)
# print(Y.value_counts())
# data["long"] = data['longitude'].apply(lambda x: round(x))
# print(data['long'].value_counts())
# data["lat"] = data['latitude'].apply(lambda x: round(x))
# print(data['lat'].value_counts())
# data['Area'] = data['long'].map(str) + '-' + data['lat'].map(str)
# data = data.drop(columns='long', axis=1)
# data = data.drop(columns='lat', axis=1)
# print(data['Area'].value_counts())

# data = data.drop(columns='Area', axis=1)

# ********* unwanted trial converting latitude and longitude to area wise split


print(data.value_counts())

# dropping latitude / longitude and are as median house cost is given directly
data = data.drop(columns='latitude', axis=1)
data = data.drop(columns='longitude', axis=1)

# plotting heatmap graph
correlation = data.corr(numeric_only=True)
sns.heatmap(correlation, annot=True)
plt.show()

# getting correlation values of median house value
print(correlation['median_house_value'])

# plotting bar graph
sns.histplot(data['median_house_value'])
plt.show()

# bar graph shows its right skewed

X = data.drop(columns='median_house_value', axis=1)
Y = data['median_house_value']


# splitting data to 80% train and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=10)

# getting shape of actual, train and test data
print(X.shape, X_train.shape, X_test.shape)
