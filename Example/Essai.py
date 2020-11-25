import pandas as pd  
import numpy as np  

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("/Users/Sebastien/Documents/3A/Optimisation/Example/bill_authentication.csv") 

print(dataset.head())

##### Preparation des data pour entrainement ######

X = dataset.iloc[:, 0:4].values  # Separation des datas des attributs
y = dataset.iloc[:, 4].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # preparation des echantillons

##### Scaling des datas #####

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

###### Training the algorithm #####

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  


print(y_pred)


###### Evaluation of algorithm #####

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))  







