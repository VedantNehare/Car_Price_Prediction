# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 19:25:01 2023

@author: Vedant Nehare
"""
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#Importing the dataset from the local storage

data =  pd.read_csv('D:/car data.csv')


x = data[[ 'Year', 'Present_Price','Kms_Driven']]  # Add more features as needed

y = data['Selling_Price']




# Split the data into training and testing sets (80% train, 20% test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#Making model with the training datasets

model =LinearRegression()
model.fit(x_train.values,y_train.values)

# predicting the output on the baisi of the value which is passed in the prediction

y_pred = model.predict(x_test)


Year = 2017
Present_Price = 9.85
Kms_Driven= 6900
price=model.predict([[Year,Present_Price,Kms_Driven]])
print("Predicted price for Year",Year,",present Price",Present_Price,"and Kms_Driven",Kms_Driven, "Predicted Price is",price[0])

