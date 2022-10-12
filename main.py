import joblib
import numpy as np
import matplotlib.pyplot
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
df= pandas.read_csv("C:\\Users\\charan\\OneDrive\\Desktop\\Toddler Autism dataset July 2018.csv")
print(df)
#X= df.drop(df.columns[[11, 12]],1,inplace = True)
x= df.iloc[:,:11]
#x= df.drop["age","relation"]
y= df.iloc[:,-1:]
print(x)
from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain,ytest = train_test_split(x,y,test_size=0.5,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xt = sc.fit_transform(xtrain)
xtr = sc.transform(xtest)
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
classifier.fit(xtrain, ytrain)
ypredict = classifier.predict(xtest)
joblib.dump(classifier, "./random_forest.joblib")
load = joblib.load("./random_forest.joblib")
#classifier.save('toddler.h5')
from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest, ypredict)
print("Random Forest Classification's Accuracy :", acc)
rel = load.predict([[0 ,0 ,0 ,0 ,0 ,0  ,0, 0 ,0, 0, 1]])
print("res=",rel)