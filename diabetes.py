import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection and analysis:
    #PIMA diabetes dataset
    #loading the diabetes dataset to a pandas dataframe
    
diabetes_dataset=pd.read_csv("C:/Users/Sakthi/Desktop/MLproject/diabetes.csv")

#printing 5 rows of dataset
diabetes_dataset.head()

#no.of.rows and columns in the dataset

diabetes_dataset.shape

#getting the statestical measure of the data.
diabetes_dataset.describe()

diabetes_dataset["Outcome"].value_counts()

#separating the data and labels.
x=diabetes_dataset.drop(columns="Outcome",axis=1)
y=diabetes_dataset["Outcome"]

print(x)
print(y)

#datastandardization
Scaler=StandardScaler()
Scaler.fit(x)

#satandardscaler(copy=time,with_mean=true,with_std=true)

standardized_data=Scaler.transform(x)
print(standardized_data)

x=standardized_data
y=diabetes_dataset["Outcome"]

print(x)
print(y)

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size= 0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#train the model:
classifier=svm.SVC(kernel="linear")

#training the svm classifier
classifier.fit(x_train,y_train)

#model evaluation
#accuracy score
#accuracy score on training data
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of the training data:",training_data_accuracy)

#accuracy score on training data
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of the test data:",test_data_accuracy)

#making predective system
input_data=("5,116,74,0,0,25.6,0.201,30")

#changing input data to numpy array

input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predecting for one instance

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data=Scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0);
    print("The person is not diabetics")
else:
    print("The person not diabetics")
