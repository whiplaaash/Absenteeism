#!/usr/bin/env python
# coding: utf-8

# # Absenteeism module

# In[1]:


# We reorganize all ingredients in the previous file:


# In[5]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# In[6]:


# CustomScaler class:

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        return self
    
    def transform(self, X, y = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]


# In[40]:


# New class to predict new data:

class absenteeism_model():
    
    def __init__(self, model_file, scaler_file):
        # read the "model_file" and "scaler_file" which were saved in the previous file:
        with open("model", "rb") as model_file, open("scaler", "rb") as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    
    # Take a data file (*.csv) and preprocess it in the same way as in previous file:
    def load_and_clean_data(self, data_file):
        
        # Import the data:
        df = pd.read_csv(data_file, delimiter = ",")
        
        # Store the data in a different file for later use:
        self.df_with_predictions = df.copy()
        
        # Drop the "ID" column:
        df = df.drop(["ID"], axis = 1)
        
        # To presrve the code we have created in the previous file we will add a column with "NaN" strings:
        df["Absenteeism Time in Hours"] = "NaN"
        
        # Create a new dataframe with all "reason" columns:
        reason_columns = pd.get_dummies(df["Reason for Absence"], drop_first = True)
        
        # Split "reason" columns into four groups:
        reason_type_1 = reason_columns.loc[:,1:14].max(axis = 1)
        reason_type_2 = reason_columns.loc[:,15:17].max(axis = 1)
        reason_type_3 = reason_columns.loc[:,18:21].max(axis = 1)
        reason_type_4 = reason_columns.loc[:,22:].max(axis = 1)
        
        # To avoid multicollinearity, drop the column "Reason for Absence":
        df = df.drop(["Reason for Absence"], axis =1)
        
        # Add new columns to the original df:
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis =1)
        
        # Assign names to newly added columns:
        # df.columns.values
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 
                        'Reason_3', 'Reason_4']
        
        df.columns = column_names
        
        # Reorder the columns:
        reordered_column_names = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 
                                  'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 
                                  'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[reordered_column_names]
        
        # Convert the "Date" column into datetime:
        df["Date"] = pd.to_datetime(df["Date"], format = "%d/%m/%Y")
        
        # Create the list of month values retrieved from the "Date" column:
        list_months = []
        for i in range(len(df["Date"])):
            list_months.append(df["Date"][i].month)
            
        # Insert the value in the new column in df, called "Month value":
        df["Month value"] = list_months
        
        # Create a new feature called "Day of the week":
        df["Day of the week"] = df["Date"].apply(lambda x: x.weekday())
        
        # Drop the "Date" column:
        df = df.drop(["Date"], axis = 1)
        
        # Reorder the columns:
        # df.columns.values
        
        reordered_column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                                      'Month value', 'Day of the week', 'Transportation Expense', 
                                      'Distance to Work', 'Age','Daily Work Load Average', 'Body Mass Index', 
                                      'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[reordered_column_names_upd]
        
        # Map "Education" variables; the result is a dummy:
        df["Education"] = df["Education"].map({1:0, 2:1, 3:1, 4:1})
        
        # Replace the "NaN" values by 0:
        df = df.fillna(value = 0)
        
        # Drop the original column "Absenteeism Time in Hours":
        df = df.drop(["Absenteeism Time in Hours"], axis = 1)
        
        # Drop the variables we decided we do not need:
        df = df.drop([ "Day of the week", "Distance to Work", "Daily Work Load Average"], axis = 1)
        
        # Make a copy of preprocessed data and keep it as a checkpoint:
        self.preprocessed_data = df.copy()
        
        # Apply the scaler from the first class:
        self.data = self.scaler.transform(df)
    
    # A function which outputs the probability of an observation being 1:
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data[:, 1])
            return pred
        
    # A function that outputs 0 or 1 based on our model:
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    # A function which adds predicted outputs and probabilities as new columns to our df:
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data["Probability"] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data["Prediction"] = self.reg.predict(self.data)
            return self.preprocessed_data
            
            
            
# This module is created and we download it to the same folder as other files as "absenteeism_module.py"

# At this moment the module should be downloaded as *.py!!!!!!!!!!!!


# In[ ]:




