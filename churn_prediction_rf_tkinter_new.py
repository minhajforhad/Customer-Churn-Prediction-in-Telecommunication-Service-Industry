# Importing the necessary packages 
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
from feature_engine import categorical_encoders as ce
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue2', relief = 'raised')
canvas1.pack()

#Loading the input data
def getCSV ():
    print("Loading input data")
    global input_df
    import_file_path = filedialog.askopenfilename()
    input_df = pd.read_csv (import_file_path, verbose = True)
    print("Some sample records from the imported file\n")
    print (input_df.head())
	
def predict():  
    print("\nRemoving the unnecessary columns from the input data frame")
    # let's remove the unnecessary columns from the data set to input to the prediction model
    necessary_cols = ['Customer ID', 'Age', 'Married', 'Number of Dependents', 'Referred a Friend','Number of Referrals', 'Tenure in Months', 'Offer', 'Internet Service', 'Avg Monthly GB Download', 'Online Security', 'Online Backup', 'Premium Tech Support', 'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method', 'Monthly Charge', 'Total Long Distance Charges', 'Total Revenue', 'CLTV']
    print("\nRemoving the unnecessary columns from the input data frame")
    input_features = input_df[necessary_cols]
    # Let's encode the binary (Yes/No) features 
    print("Encoding the binary yes/no columns")
    binary_cols = ['Married', 'Referred a Friend', 'Internet Service', 'Online Security', 'Online Backup', 'Premium Tech Support', 'Unlimited Data', 'Paperless Billing']
    input_features2 = input_features.copy()
    for col in binary_cols:         
        input_features2[col].replace(to_replace='Yes', value=1, inplace=True)
        input_features2[col].replace(to_replace='No',  value=0, inplace=True)
        # Replacing 'None' as 'No Offer' as it is a naturlal category of the variable 'Offer' which means no marketing offer was accepted by the customer
    input_features2['Offer'].replace(to_replace='None', value='No offer', inplace=True)
	
	# Let's transform the categorical features into numerical using the pickled transformer
 
    print("Transforming the categorical features into numeric")
    cat_encoder = pickle.load(open('cat_encoder_new2.pkl','rb'))
    input_features_t= cat_encoder.transform(input_features2)
    input_features_final = input_features_t.drop(columns = ['Customer ID'])
    print("Predicting the churn class and probability using the trained model saved as pickle")
    # Now let's predict using the saved model
    model_rf = pickle.load(open('model_rf_imb.pkl','rb'))
    output_predicted_class = model_rf.predict(input_features_final)
    output_predicted_prob = model_rf.predict_proba(input_features_final)

    output_predicted_class_series = pd.Series(output_predicted_class)
    print("\nSummary of the predicted class") 
    print(output_predicted_class_series.value_counts())
    global output_df
    global output_file_name
    output_df = input_df.copy()
    output_df['Predicted Churn Label'] = output_predicted_class
    output_df['Predicted Churn Probability'] = output_predicted_prob[:,1]
    output_df['Predicted Churn Label'].replace(to_replace=1, value='Yes', inplace=True)
    output_df['Predicted Churn Label'].replace(to_replace=0, value='No', inplace=True)
    #output_df['Predicted Churn Label'].value_counts()
    print("\nSaving the output file with predicted churn class and probability")
    #output_df.to_csv('churn\\output\\churn_output.csv', index = False)
    global output_file_name
    output_file_name = 'churn_output_'+datetime.now().strftime('%Y%m%d%H%M%S')+'.csv'
    #output_df.to_csv('churn\\output\\'+ output_file_name, index = False)

    # We can target the customers who show highest churn propensity
    print("\nTop 10 customers having the highest churn propensity:\n")
    print(output_df[['Customer ID', 'Predicted Churn Label', 'Predicted Churn Probability']].sort_values(by='Predicted Churn Probability', ascending = False).head(10))
    

def exportCSV ():    
    export_file_path = filedialog.asksaveasfilename(confirmoverwrite=True, title="Save the file",initialfile = output_file_name, defaultextension='.csv')
    output_df.to_csv(export_file_path, index = False, header=True)

browseButton_CSV = tk.Button(text="      Import input data file     ", command=getCSV, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 90, window=browseButton_CSV)
predictButton_CSV = tk.Button(text="      Make churn prediction     ", command=predict, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 130, window=predictButton_CSV)
saveAsButton_CSV = tk.Button(text='Export output file', command=exportCSV, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 170, window=saveAsButton_CSV)

tk.Label(root, text="Created by Minhaj", fg = "blue", bg = "yellow", font = "Verdana 10 bold").pack()
root.title("Churn Prediction App")
root.mainloop()

