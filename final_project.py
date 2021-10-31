#!/usr/bin/env python
# coding: utf-8

import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache(allow_output_mutation=True)

# Read in Dataset
def load_data(n_rows):
    df = pd.read_csv("LaundryData.csv")
    df.columns = df.columns.str.upper()
    df.set_index('NO',inplace=True)
    return df

df = load_data(807)

# Streamlit 
html_temp = """
<div style ="background-color:white;padding:3.5px">
<h1 style="color:black;text-align:center;">TDS 3301 Data Mining</h1>
</div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Data Pre-Processing 
st.title("Data Pre-Processing")

# Data Cleaning
st.markdown("### Data Cleaning")
st.write("The detected missing values in this dataset is replaced by filling in the mode of each features."
" The data types of each column and the possible duplication of data have been checked to ensure the consistency of the dataset. "
"  Also, the date and time columns have been discretized into a single column with the combination of date and time.")

if st.checkbox("Dealing with Missing Values (Before)"):
    img = Image.open('before clean.jpg')
    st.image(img, width = 150)
pass 

miss_list = ['RACE','GENDER','BODY_SIZE','AGE_RANGE','WITH_KIDS','KIDS_CATEGORY','BASKET_SIZE','BASKET_COLOUR','ATTIRE','SHIRT_COLOUR','SHIRT_TYPE','PANTS_COLOUR','PANTS_TYPE','WASH_ITEM']

for i in miss_list:
    df[i] = df[i].fillna(df[i].mode()[0])

if st.checkbox("Dealing with Missing Values (After)"):
    img = Image.open('after clean.jpg')
    st.image(img, width = 150)
pass 

st.write("")

# Inconsistent Data Types 
if st.checkbox("Dealing with Inconsistent Data Types (Before）"):
   img = Image.open('before data type.jpg')
   st.image(img, width = 200)
pass 

df['AGE_RANGE'] = df['AGE_RANGE'].astype('int64')
df['WASHER_NO'] = df['WASHER_NO'].astype('str')
df['DRYER_NO'] = df['DRYER_NO'].astype('str')

if st.checkbox("Dealing with Inconsistent Data Types (After)"):
   img = Image.open('after data type.jpg')
   st.image(img, width = 200)
pass 

# Data Discretization
st.markdown("### Data Discretization")

df['TIME'] = df['TIME'].str.replace(';',':')
df['DATE_TIME'] =  pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%d/%m/%Y %H:%M:%S')
dff = df.copy()
dff.drop(['DATE','TIME'],axis='columns',inplace=True) 

if st.checkbox("Dealing with Data Representation"):
    st.write(dff)
pass
 
# Data Duplication
st.markdown("### Data Duplication")
if st.checkbox("Dealing with Duplicated Data"):
   duplicate = sum(dff.duplicated())
   st.write("The number of duplicated data in this dataset is",duplicate)
pass

# Label Encoding 
st.markdown("### Label Encoding")
st.write("Lebel Encoding is a method where labels are converted into the machine-reable form. "
"In this project, a new dataset that undergoes label encoding is performed in numeric form.")

copy = dff.copy()
df_clean = copy.apply(LabelEncoder().fit_transform)
st.write(df_clean)

st.write("")

# Correlation 
st.markdown("### Correlation")
if st.checkbox("Correlation Plot"):
   img = Image.open('correlation.jpg')
   st.image(img)
   st.write("Based on the correlation plot, the 'WITH_KIDS' feature and the 'KIDS_CATEGORY' shows a strong correlation within each other at 0.65.")
pass 
 
# EDA
st.title("Exploratory Data Analysis")

if st.checkbox("Question 1"):
    st.write("1. Which age groups are the most visiting customers to the laundry shop?")
    img = Image.open('q1.jpg')
    st.image(img)
    st.write("Based on the bar plot, it has found out that the customers in the range of 40 to 50 years old are more often visit to the laundry shop."
    " The following groups are the range of 30 to 40 years old and the least often visiting groups is the customers in the range of 20 to 30 years old.")
pass 

if st.checkbox("Question 2"):
    st.write("2. Which washer and dryer number will be used to wash normal clothes more often?")
    img = Image.open('q2.jpg')
    st.image(img)
    img1 = Image.open('q2b.jpg')
    st.image(img1)
    st.write("Based on the plot, the washer 3 has the highest usage to wash the blankets and the normal clothes while the washer 5 has the least usage in washing blankets and the washer 4 has the lowest usage in washing clothes among the four washers."
    " For the usage of dryer, the dryer 7 has the most usage in drying clothes while the dryer 8 has the least usage. For drying blankets as well, the dryer 7 has the most usage compared to the dryer 9.")

if st.checkbox("Question 3"):
    st.write("3. Does the majority of big basket size contain blankets while the majority of small basket size contains clothes?")
    img = Image.open('q3.jpg')
    st.image(img)
    st.write("Based on the confusion matrix, it shows that the majority of the customers use big baskets to place their clothes, "
    " whereas data also shows that customers seldom use small baskets to place their blankets. "
    " Besides that, data shows that customers rarely bring small baskets to the laundry shop compared to the big basket as the clothes and blankets "
    " placed in big baskets are higher than small baskets.")
pass 

if st.checkbox("Question 4"):
    st.write("4. Does the race affect the basket size?")
    img = Image.open('q4.jpg')
    st.image(img)
    st.write("Based on the confusion matrix, it shows that the locals will visit the laundry shop more often compared to the foreigners. "
    "Race does not affect the basket size used by the customer as from the result shown the majority of the customer choose to use a big basket "
    "compared to the small basket.")
pass 

# Feature Selection 
st.title("Feature Selection")
st.write("There are several method that used in the feature selection which included Boruta, Recursive Feature Elimination (RFE) and the Chi-Squared test."
" The top 10 features of each method have been listed out as comparison. Then, the best features within each method have been selected for further analysis.")

if st.checkbox("Boruta"):    
    img = Image.open('boruta.jpg')
    st.image(img)
pass

if st.checkbox("RFE"):
    img = Image.open('rfe.jpg')
    st.image(img)
pass 

# Chi-squared 
if st.checkbox("Chi-Squared test"):
    img = Image.open('k2.jpg')
    st.image(img)
pass 

st.markdown("### Best Features within Boruta, RFE and Chi-Squared test")
st.write("There is a total of 6 best features as:")
st.write("- RACE")
st.write("- PANTS_COLOUR")
st.write("- BASKET_SIZE")
st.write("- BASKET_COLOUR")
st.write("- SHIRT_COLOUR")
st.write("- ATTIRE")

st.title("Machine Learning Techniques")
# Association Rule Mining 

st.markdown("### Association Rule Mining")
st.write("Association rule mining is used to find the frequent patterns, correlations, associations, or causal structures among the items from the data."
" In this part, the apriori algorithm with the association rules in identifying the amount of the associated items amoung the dataset."
" By using the apriori algorithm, it is able to identify that there are a total of 52 frequent items. There is also a total of 114 rules by applying the association rules algorithm.")
if st.checkbox("Apriori Algorithm"):
    img = Image.open('apriori.jpg')
    st.image(img)
    st.write("The top 20 of the frequent items is shown by applying the apriori algorithm. ")
pass 

if st.checkbox("Association Rules"):
    img = Image.open('arm.jpg')
    st.image(img)
    st.write("The top 20 of the rules is shown by applying the association rules algorithm.")
    st.write("")
    st.write("Based on the result, the big basket size, short sleeve shirt type and casual attire being the candidates with the highest lift scores of 1.09."
    " It means that most of the customers come with the casual attire of short sleeves with the big baskets to the laundry shop.")
pass 

# K-means clustering
st.markdown("### K-means Clustering")
st.write("The K-Means clustering chosen as the clustering technique as it is one of the more versatile techniques that can handle in most of the situation."
"The best features that selected from the comparison within different methods have used for this K-Means clustering technique. The best features have undergo label encoding for further analysis.")

if st.checkbox("K-means Plot"):
    img = Image.open('kmeans.jpg')
    st.image(img)
    st.write("Based on the elbow criterion, the ideal k values to choose for is four.")
pass 

if st.checkbox("Silhouette Score"):
    st.write("The silhouette score for the K-Means clustering of k value is 0.21 as all four clusters are above the average silhouette score and the fluctuations"
    " are more uniform in the cluster four as well.")
pass 

# Classification 
st.title("Classification Model")
st.write("The strong features have been selected for model implementation. The dataset has split into 70% train set and 30% test set with the random state of 10."
" Also, the dataset has undergo SMOTE to prevent the oversampling problem. Therefore, the overall classification model included the dataset with SMOTE and without SMOTE.")

st.write("There are five types of classfication model have been applied as :")
st.write("- K-Nearest Neighbors (KNN) Classifier")
st.write("- Logistic Regression")
st.write("- Naive Bayes")
st.write("- Decision Tree Classifier")
st.write("- Support Vector Classifier (SVC)")

st.markdown("## Overview for Classification Model")
st.markdown("### Receiver Operating Characteristic (ROC) Curve")
if st.checkbox("Overview ROC curve without SMOTE"):
    img = Image.open('roc.jpg')
    st.image(img)
    st.write("")
    st.write("Based on the ROC curve graph, the overall analysis on Logistic Regression has the least performance, which is AUC at 62%, followed by AUC of Naive Bayes and KNN both are 65%. "
    "Then for the next algorithm Support Vector Classifier has AUC 67%. The most outperformed model among the classification models is Decision Tree Classifier which has an AUC of 76%.")
pass 

if st.checkbox("Overview ROC curve with SMOTE"):
    img = Image.open('roc-smote.jpg')
    st.image(img)
    st.write("")
    st.write("Based on the plot, the lowest ROC curve is still the logistic regression. "
    " It has the lowest AUC which is 62%. Then follow up with the model which has the nearly same AUC, which is Naive Bayes 65%, "
    "KNN 66%, and SVC 67%. Decision Tree Classifiers still have the highest AUC which is 73% ""among the other models. ")
pass 

st.markdown("### Precision-Recall Curve")
if st.checkbox("Overview Precision-Recall Curve without SMOTE"):
    img = Image.open('pcr.jpg')
    st.image(img)
    st.write("")
    st.write("For the precision-recall curve, the Logistic Regression has the least overall precision among the models. "
    "KNN, Naive Bayes and Support Vector Classifier have quite similar precision recall curves. For the decision tree classifier, "
    "it is considered best as it has a good and more stable precision value over recall compared to the other models.")
pass 

if st.checkbox("Overview Precision-Recall Curve with SMOTE"):
    img = Image.open('pcr-smote.jpg')
    st.image(img)
    st.write("")
    st.write("For the precision-recall curve with SMOTE applied, the Logistic regression still has the least overall precision among the models. "
    "The KNN, Naive Bayes and Support Vector Classifier have quite similar precision recall curves. "
    "Besides that there is a significant drop of precision for KNN, Naive Bayes, Support Vector Classifier and Logistic Regression when the recall reaches around 0.6.")
    st.write("")
    st.write("As conclude, the Decision Tree Classifier is the best classification model as it contains the highest accuracy score among the other models."
    " The Decision Tree Classifier is also the most outperformed model as it consists of the highest AUC score and precision score.")
pass 

st.title("Regression Model")
st.write("For the regression model as well, the best features are selected for the model implementation. The dataset has distribute into SMOTE and without SMOTE."
" The test train split has applied to random split the dataset into 70% train set and 30% test set.")

st.write("There are two types of regression model as:")
st.write("- Linear Regression")
st.write("- Decision Tree Regressor")

st.markdown("### Overview for Regression Model")
if st.checkbox("Regression Model without SMOTE"):
    img = Image.open('reg.jpg')
    st.image(img)
    st.write("Based on the regression plot without SMOTE, The red plot represents the linear regression and the blue plot represents the decision tree regressor. "
    " The accuracy score for both models is quite low. The accuracy score for decision tree regressor is 17.1% which is slightly higher than linear regression which contains of 5.2%."
    " The overall accuracy score for both model is low as it may be the predicting dataset is in categorical data form.")
pass 

if st.checkbox("Regression Model with SMOTE"):
    img = Image.open('reg-smote.jpg')
    st.image(img)
    st.write("With the application of  SMOTE to the dataset, the result of the model looks similar to the one when SMOTE is not applied.  Decision Tree Regressor still has a higher accuracy compared to linear regression.")
pass 
