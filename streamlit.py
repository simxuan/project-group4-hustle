

import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 

import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 

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
   df_corr = df_clean.copy() 
   corr = df_corr.corr()
   sns.set(rc={'figure.figsize':(10,10)})
   ax = sns.heatmap(corr, vmax=.8, square=True, annot=True, fmt= '.2f',annot_kws={'size': 8}, cmap=sns.color_palette("Reds"))
   st.pyplot(ax=ax)
   st.write("Based on the correlation plot, the 'WITH_KIDS' feature and the 'KIDS_CATEGORY' shows a strong correlation within each other at 0.65.")
pass 
 
# EDA
st.title("Exploratory Data Analysis")

if st.checkbox("Question 1"):
    st.write("1. Which age groups are the most visiting customers to the laundry shop?")
    df2 = dff.copy()
    bins = [0,10,20,30,40,50,60]
    df2['AGE_GROUP'] = pd.cut(df2['AGE_RANGE'],bins=bins)
    sns.set(rc={'figure.figsize':(10,10)})
    plt.xlabel("Range of Age Group")
    ax = df2['AGE_GROUP'].value_counts().plot(kind='bar')
    st.pyplot(ax=ax)
    st.write("Based on the bar plot, it has found out that the customers in the range of 40 to 50 years old are more often visit to the laundry shop."
    " The following groups are the range of 30 to 40 years old and the least often visiting groups is the customers in the range of 20 to 30 years old.")
pass 

if st.checkbox("Question 2"):
    st.write("2. Which washer and dryer number will be used to wash normal clothes more often?")
    def laundry_plot(df,by,y,stack=False,sort=False,kind='bar'):
        pivot = df.groupby(by)[y].count().unstack(y)
        pivot = pivot.sort_values(by=pivot.columns.to_list(),ascending=False)
        ax = pivot.plot(kind=kind,stacked=stack,figsize =(8,8))
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    sns.set(rc={'figure.figsize':(10,10)})
    ax = laundry_plot(df,['WASHER_NO','WASH_ITEM'],'WASH_ITEM')
    plt.ylabel("Total amount of Wash Item")
    st.pyplot(ax=ax)
    sns.set(rc={'figure.figsize':(10,10)})
    axx = laundry_plot(df,['DRYER_NO','WASH_ITEM'],'WASH_ITEM')
    plt.ylabel("Total amount of Wash Item")
    st.pyplot(ax=axx)
    st.write("Based on the plot, the washer 3 has the highest usage to wash the blankets and the normal clothes while the washer 5 has the least usage in washing blankets and the washer 4 has the lowest usage in washing clothes among the four washers."
    " For the usage of dryer, the dryer 7 has the most usage in drying clothes while the dryer 8 has the least usage. For drying blankets as well, the dryer 7 has the most usage compared to the dryer 9.")

if st.checkbox("Question 3"):
    st.write("3. Does a big basket size contain blankets while a small basket size contains clothes?")
    def laundry_heatmap(df,by,count_att,y):
        df = df.groupby(by)[count_att].count().unstack(y)
        fig = plt.figure(figsize=(5,5))
        heatmap = sns.heatmap(df,annot =  df.values ,fmt='g')
        heatmap.set_xticklabels(["Blanket", "Clothes"])
        heatmap.set_yticklabels(["Big", "Small"])
    sns.set(rc={'figure.figsize':(5,5)})
    ax = laundry_heatmap(df,by=['BASKET_SIZE','WASH_ITEM'],count_att='WASHER_NO',y='WASH_ITEM')
    st.pyplot(ax=ax)
    st.write("Based on the confusion matrix, it shows that the majority of the customers use big baskets to place their clothes, "
    " whereas data also shows that customers seldom use small baskets to place their blankets. "
    " Besides that, data shows that customers rarely bring small baskets to the laundry shop compared to the big basket as the clothes and blankets "
    " placed in big baskets are higher than small baskets.")
pass 

if st.checkbox("Question 4"):
    st.write("4. Does the race affect the basket size?")
    def laundry_heatmap2(df,by,count_att,y):
        df = df.groupby(by)[count_att].count().unstack(y)
        fig = plt.figure(figsize=(5,5))
        heatmap = sns.heatmap(df,annot =  df.values ,fmt='g')
        heatmap.set_xticklabels(["Chinese", "Foreigners","Indian","Malay"])
        heatmap.set_yticklabels(["Big", "Small"])
    sns.set(rc={'figure.figsize':(5,5)})
    ax = laundry_heatmap2(df,by=['BASKET_SIZE','RACE'],count_att='BASKET_SIZE',y='RACE')
    st.pyplot(ax=ax)
    st.write("Based on the confusion matrix, it shows that the locals will visit the laundry shop more often compared to the foreigners. "
    "Race does not affect the basket size used by the customer as from the result shown the majority of the customer choose to use a big basket "
    "compared to the small basket.")
pass 

# Feature Selection 
st.title("Feature Selection")
st.write("There are several method that used in the feature selection which included Boruta, Recursive Feature Elimination (RFE) and the Chi-Squared test."
" The top 10 features of each method have been listed out as comparison. Then, the best features within each method have been selected for further analysis.")

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

df3 = df_clean.copy()
X = df3.drop(columns = ['DATE_TIME'],axis=1)
y = df3['DATE_TIME']
colnames = X.columns

# Boruta 
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)
feat_selector.fit(X.values,y.values.ravel())
boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values('Score',ascending=False)

if st.checkbox("Boruta"):    
    sns.set(rc={'figure.figsize':(10,10)})
    sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:], kind = "bar", height=14, aspect=1.9, palette='coolwarm')
    st.pyplot(ax=sns_boruta_plot)
pass

# RFE
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5,random_state=5)
rf.fit(X,y)
rfe = RFECV(rf,min_features_to_select=1,cv=5)
rfe.fit(X,y)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

if st.checkbox("RFE"):
    sns.set(rc={'figure.figsize':(10,10)})
    sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:], kind = "bar", height=14, aspect=1.9, palette='coolwarm')
    #total_feature = rfe_score.shape[0]
    st.pyplot(ax=sns_rfe_plot)
pass 

# Chi-squared 
select_feature = SelectKBest(chi2, k=10).fit(X, y) # top 10
kbest = select_feature.get_support(indices=True)
new_feature = X.iloc[:,kbest]
#for i in new_feature.columns:
        #st.write(i)
score = pd.DataFrame(select_feature.scores_ , X.columns).sort_values(0, ascending=False).reset_index().rename(columns={0:"score", "index":"Feature"})

if st.checkbox("Chi-Squared test"):
    sns.set(rc={'figure.figsize':(5,5)})
    ax = sns.barplot(data=score, x="score", y="Feature", color="b")
    st.pyplot(ax=ax)
pass 

st.markdown("### Best Features within Boruta, RFE and Chi-Squared test")

boruta_feature = set(boruta_score[:10]['Features'])
rfe_feature = set(rfe_score[:10]['Features'])
chi_feature = set(new_feature.columns)

boruta_rfe = boruta_feature.intersection(rfe_feature)
boruta_rfe_chi = boruta_rfe.intersection(chi_feature)
st.write(boruta_rfe_chi)

st.title("Machine Learning Techniques")
# Association Rule Mining 
high_corr_attri = list(df.columns)
to_remove = ['DATE_TIME','AGE_RANGE','WITH_KIDS']
for i in to_remove:
    high_corr_attri.remove(i)
temp = df.copy()
temp = temp[high_corr_attri]
dummy= pd.get_dummies(temp)

frequent_itemsets = apriori(dummy,min_support=0.5, use_colnames=True, max_len=3)
frequent_itemsets.sort_values(by='support',ascending=False,ignore_index=True,inplace=True)

st.markdown("### Association Rule Mining")
st.write("Association rule mining is used to find the frequent patterns, correlations, associations, or causal structures among the items from the data."
" In this part, the apriori algorithm with the association rules in identifying the amount of the associated items amoung the dataset."
" By using the apriori algorithm, it is able to identify that there are a total of 52 frequent items. There is also a total of 114 rules by applying the association rules algorithm.")
if st.checkbox("Apriori Algorithm"):
    st.write(frequent_itemsets.head(20))
    st.write("The top 20 of the frequent items is shown by applying the apriori algorithm. ")
pass 

if st.checkbox("Association Rules"):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    arm = rules.head(20).sort_values(by='lift',ascending=False)
    st.write(arm)
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
    distortions = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    st.pyplot()
    st.write("Based on the elbow criterion, the ideal k values to choose for is four.")
pass 

if st.checkbox("Silhouette Score"):
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    label=km.predict(X)
    score = silhouette_score(X, label)
    st.write("The silhouette score for the K-Means clustering of k value is", score, " as all four clusters are above the average silhouette score and the fluctuations"
    " are more uniform in the cluster four as well.")
pass 

# Train Test Split
X = df3.drop(['DATE_TIME','SHIRT_TYPE','WASH_ITEM','SPECTACLES','GENDER','BODY_SIZE','PANTS_TYPE','DRYER_NO','KIDS_CATEGORY','AGE_RANGE','WITH_KIDS','WASHER_NO'],axis=1)
y = df3.GENDER

# Without SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

# With SMOTE 
smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
X_res, y_res = smt.fit_resample(X, y)
X_train_smt, X_test_smt, y_train_smt, y_test_smt = train_test_split(X_res, y_res, test_size=0.30, random_state=10)

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

# KNN 
# Without SMOTE 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
prob_KNN = knn.predict_proba(X_test)
prob_KNN = prob_KNN[:,1]
auc_KNN = roc_auc_score(y_test, prob_KNN)
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test, prob_KNN) 
prec_KNN, rec_KNN, threshold_KNN = metrics.precision_recall_curve(y_test, prob_KNN)

# With SMOTE 
knn_smt = KNeighborsClassifier(n_neighbors=3)
knn_smt.fit(X_train_smt, y_train_smt)
y_pred_smt = knn_smt.predict(X_test_smt)
prob_KNN_smt = knn_smt.predict_proba(X_test_smt)
prob_KNN_smt = prob_KNN_smt[:,1]
auc_KNN_smt = roc_auc_score(y_test_smt, prob_KNN_smt)
fpr_KNN_smt, tpr_KNN_smt, thresholds_KNN_smt = metrics.roc_curve(y_test_smt, prob_KNN_smt) 
prec_KNN_smt, rec_KNN_smt, threshold_KNN_smt = metrics.precision_recall_curve(y_test_smt, prob_KNN_smt)

# Logistic Regression
# Without SMOTE 
logreg = LogisticRegression(solver='liblinear',max_iter=400, fit_intercept=True, random_state=10)
logreg.fit(X_train, y_train) 
y_pred = logreg.predict(X_test)
prob_LR = logreg.predict_proba(X_test)
prob_LR = prob_LR[:,1]
auc_LR = roc_auc_score(y_test, prob_LR)
fpr_LR, tpr_LR, thresholds_LR = metrics.roc_curve(y_test, prob_LR) 
prec_LR, rec_LR, threshold_LR = metrics.precision_recall_curve(y_test, prob_LR)

# With SMOTE 
logreg_smt = LogisticRegression(solver='liblinear',max_iter=400, fit_intercept=True, random_state=10)
logreg_smt.fit(X_train_smt, y_train_smt) 
y_pred_smt = logreg_smt.predict(X_test_smt)
prob_LR_smt = logreg_smt.predict_proba(X_test_smt)
prob_LR_smt = prob_LR_smt[:,1]
auc_LR_smt = roc_auc_score(y_test_smt, prob_LR_smt)
fpr_LR_smt, tpr_LR_smt, thresholds_LR_smt = metrics.roc_curve(y_test_smt, prob_LR_smt) 
prec_LR_smt, rec_LR_smt, threshold_LR_smt = metrics.precision_recall_curve(y_test_smt, prob_LR_smt)

# Naive Bayes 
# Without SMOTE 
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
prob_NB = nb.predict_proba(X_test)
prob_NB = prob_NB[:,1]
auc_NB = roc_auc_score(y_test, prob_NB)
fpr_NB, tpr_NB, thresholds_NB = metrics.roc_curve(y_test, prob_NB) 
prec_NB, rec_NB, threshold_NB = metrics.precision_recall_curve(y_test, prob_NB)

# With SMOTE 
nb_smt = GaussianNB()
nb_smt.fit(X_train_smt, y_train_smt)
y_pred_smt = nb_smt.predict(X_test_smt)
prob_NB_smt = nb_smt.predict_proba(X_test_smt)
prob_NB_smt = prob_NB_smt[:,1]
auc_NB_smt = roc_auc_score(y_test_smt, prob_NB_smt)
fpr_NB_smt, tpr_NB_smt, thresholds_NB_smt = metrics.roc_curve(y_test_smt, prob_NB_smt) 
prec_NB_smt, rec_NB_smt, threshold_NB_smt = metrics.precision_recall_curve(y_test_smt, prob_NB_smt)

# Decision Tree Classifier 
# Without SMOTE 
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=6, splitter='best', random_state=10)
dtree.get_params()
dtree = dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
prob_dtree = dtree.predict_proba(X_test)
prob_dtree = prob_dtree[:,1]
auc_dtree = roc_auc_score(y_test, prob_dtree)
fpr_dtree, tpr_dtree, thresholds_dtree = metrics.roc_curve(y_test, prob_dtree) 
prec_dtree, rec_dtree, threshold_dtree = metrics.precision_recall_curve(y_test, prob_dtree)

# With SMOTE 
dtree_smt = DecisionTreeClassifier(criterion="entropy", max_depth=6, splitter='best', random_state=10)
dtree_smt.get_params()
dtree_smt = dtree_smt.fit(X_train_smt,y_train_smt)
y_pred_smt = dtree_smt.predict(X_test_smt)
prob_dtree_smt = dtree_smt.predict_proba(X_test_smt)
prob_dtree_smt = prob_dtree_smt[:,1]
auc_dtree_smt = roc_auc_score(y_test_smt, prob_dtree_smt)
fpr_dtree_smt, tpr_dtree_smt, thresholds_dtree_smt = metrics.roc_curve(y_test_smt, prob_dtree_smt) 
prec_dtree_smt, rec_dtree_smt, threshold_dtree_smt = metrics.precision_recall_curve(y_test_smt, prob_dtree_smt)

# Support Vector Classifier 
# Without SMOTE 
svc = SVC(kernel = 'rbf', C=10, gamma =0.01, probability=True, random_state=10)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
prob_svc = svc.predict_proba(X_test)
prob_svc = prob_svc[:,1]
auc_svc = roc_auc_score(y_test, prob_svc)
fpr_svc, tpr_svc, thresholds_svc = metrics.roc_curve(y_test, prob_svc) 
prec_svc, rec_svc, threshold_svc = metrics.precision_recall_curve(y_test, prob_svc)

# With SMOTE 
svc_smt = SVC(kernel = 'rbf', C=10, gamma =0.01, probability=True, random_state=10)
svc_smt.fit(X_train_smt,y_train_smt)
y_pred_smt = svc_smt.predict(X_test_smt)
prob_svc_smt = svc_smt.predict_proba(X_test_smt)
prob_svc_smt = prob_svc_smt[:,1]
auc_svc_smt = roc_auc_score(y_test_smt, prob_svc_smt)
fpr_svc_smt, tpr_svc_smt, thresholds_svc_smt = metrics.roc_curve(y_test_smt, prob_svc_smt) 
prec_svc_smt, rec_svc_smt, threshold_svc_smt = metrics.precision_recall_curve(y_test_smt, prob_svc_smt)

st.markdown("## Overview for Classification Model")
st.markdown("### Receiver Operating Characteristic (ROC) Curve")
if st.checkbox("Overview ROC curve without SMOTE"):
    plt.plot(fpr_KNN, tpr_KNN, color='orange', label='KNN') 
    plt.plot(fpr_LR, tpr_LR, color='blue', label='Logistic Regression')  
    plt.plot(fpr_NB, tpr_NB, color='red', label='Naive Bayes') 
    plt.plot(fpr_dtree, tpr_dtree, color='black', label='Decision Tree Classifier')
    plt.plot(fpr_svc, tpr_svc, color='purple', label='Support Vector Classifier') 
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve without SMOTE')
    plt.legend()
    sns.set(rc={'figure.figsize':(10,10)})
    st.pyplot()
    st.write("")
    st.write("Based on the ROC curve graph, the overall analysis on Logistic Regression has the least performance, which is AUC at 62%, followed by AUC of Naive Bayes and KNN both are 65%. "
    "Then for the next algorithm Support Vector Classifier has AUC 67%. The most outperformed model among the classification models is Decision Tree Classifier which has an AUC of 76%.")
pass 

if st.checkbox("Overview ROC curve with SMOTE"):
    plt.plot(fpr_KNN_smt, tpr_KNN_smt, color='orange', label='KNN') 
    plt.plot(fpr_LR_smt, tpr_LR_smt, color='blue', label='Logistic Regression')  
    plt.plot(fpr_NB_smt, tpr_NB_smt, color='red', label='Naive Bayes') 
    plt.plot(fpr_dtree_smt, tpr_dtree_smt, color='black', label='Decision Tree Classifier')
    plt.plot(fpr_svc_smt, tpr_svc_smt, color='purple', label='Support Vector Classifier') 
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve with SMOTE')
    plt.legend()
    st.pyplot()
    st.write("")
    st.write("Based on the plot, the lowest ROC curve is still the logistic regression. "
    " It has the lowest AUC which is 62%. Then follow up with the model which has the nearly same AUC, which is Naive Bayes 65%, "
    "KNN 66%, and SVC 67%. Decision Tree Classifiers still have the highest AUC which is 73% ""among the other models. ")
pass 

st.markdown("### Precision-Recall Curve")
if st.checkbox("Overview Precision-Recall Curve without SMOTE"):
    plt.plot(prec_KNN, rec_KNN, color='orange', label='KNN') 
    plt.plot(prec_LR, rec_LR, color='blue', label='Logistic Regression') 
    plt.plot(prec_NB, rec_NB, color='red', label='Naive Bayes') 
    plt.plot(prec_dtree, rec_dtree, color='black', label='Decision Tree Classifier') 
    plt.plot(prec_svc, rec_svc, color='purple', label='Support Vector Classifier') 
    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve without SMOTE')
    plt.legend()
    st.pyplot()
    st.write("")
    st.write("For the precision-recall curve, the Logistic Regression has the least overall precision among the models. "
    "KNN, Naive Bayes and Support Vector Classifier have quite similar precision recall curves. For the decision tree classifier, "
    "it is considered best as it has a good and more stable precision value over recall compared to the other models.")
pass 

if st.checkbox("Overview Precision-Recall Curve with SMOTE"):
    plt.plot(prec_KNN_smt, rec_KNN_smt, color='orange', label='KNN') 
    plt.plot(prec_LR_smt, rec_LR_smt, color='blue', label='Logistic Regression') 
    plt.plot(prec_NB_smt, rec_NB_smt, color='red', label='Naive Bayes') 
    plt.plot(prec_dtree_smt, rec_dtree_smt, color='black', label='Decision Tree Classifier') 
    plt.plot(prec_svc_smt, rec_svc_smt, color='purple', label='Support Vector Classifier') 
    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with SMOTE')
    plt.legend()
    st.pyplot()
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

X = df3.drop(['DATE_TIME','SHIRT_TYPE','WASH_ITEM','SPECTACLES','GENDER','BODY_SIZE','PANTS_TYPE','DRYER_NO','KIDS_CATEGORY','AGE_RANGE','WITH_KIDS','WASHER_NO'],axis=1)
y = df3.GENDER

# Without SMOTE 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=10)

# With SMOTE
smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
X_res, y_res = smt.fit_resample(X, y)
X_train_smt, X_test_smt, y_train_smt, y_test_smt = train_test_split(X_res, y_res, test_size=0.30, random_state=10)

# Linear Regression 
# Without SMOTE 
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_LR = lm.predict(X_test)

# With SMOTE 
lm_smt = LinearRegression()
lm_smt.fit(X_train_smt, y_train_smt)
y_pred_LR_smt = lm_smt.predict(X_test_smt)

# Decision Tree Regressor 
# Without SMOTE 
dtreg = DecisionTreeRegressor(max_depth=6, criterion="mse", splitter="best")  
dtreg.fit(X_train, y_train) 
y_pred_dtreg = dtreg.predict(X_test)

# With SMOTE 
dtreg_smt = DecisionTreeRegressor(max_depth=6, criterion="mse", splitter="best")  
dtreg_smt.fit(X_train_smt, y_train_smt) 
y_pred_dtreg_smt = dtreg_smt.predict(X_test_smt)

st.markdown("### Overview for Regression Model")
if st.checkbox("Regression Model without SMOTE"):
    sns.regplot(x=y_test, y=y_pred_LR, x_jitter=.01, scatter_kws={"color": "red"}, line_kws={"color": "red"})
    sns.regplot(x=y_test, y=y_pred_dtreg, x_jitter=.01, scatter_kws={"color": "blue"}, line_kws={"color": "blue"})
    st.pyplot()
    st.write("Based on the regression plot without SMOTE, The red plot represents the linear regression and the blue plot represents the decision tree regressor. "
    " The accuracy score for both models is quite low. The accuracy score for decision tree regressor is 17.1% which is slightly higher than linear regression which contains of 5.2%."
    " The overall accuracy score for both model is low as it may be the predicting dataset is in categorical data form.")
pass 

if st.checkbox("Regression Model with SMOTE"):
    sns.regplot(x=y_test_smt, y=y_pred_LR_smt, x_jitter=.01, scatter_kws={"color": "red"}, line_kws={"color": "red"})
    sns.regplot(x=y_test_smt, y=y_pred_dtreg_smt, x_jitter=.01, scatter_kws={"color": "blue"}, line_kws={"color": "blue"})
    st.pyplot()
    st.write("With the application of  SMOTE to the dataset, the result of the model looks similar to the one when SMOTE is not applied.  Decision Tree Regressor still has a higher accuracy compared to linear regression.")
pass 
