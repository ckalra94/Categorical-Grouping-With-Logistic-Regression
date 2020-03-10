#Read the .readme file first!

# FakeData = .csv containing everything with KPI values available
# NewData = .csv containing everything except KPI values, if applicable
# FakeData is required. NewData is optional
# If NewData will be imported, "has_binary" must be set to 'No'
#
# 


#Data Creation
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math

n=5000

income=np.random.exponential(1, n)
income=income+1
income=income*5000
#plt.hist(income)
debtToIncome=np.random.normal(3.24, 3, n)
#plt.hist(debtToIncome)
propensityScore=np.random.uniform(0, 100, n)
propensityScore=list(propensityScore)
for i in range(0, n):
    propensityScore[i] = math.ceil(propensityScore[i])

gender=np.random.choice(['M', 'F'], n)
location=np.random.choice(['CA', 'WA', 'TX'], n)
account_age=np.random.choice(['New', 'Existing', 'Tenured', 'Grandfathered'], n)
prior_model_risk=np.random.choice(['Low', 'Med', 'High'], n)
default=[np.random.binomial(1, 0.33) for x in range(0,n)]

FakeData=pd.DataFrame({'default':default, 'income':income, 'prior_model_risk':prior_model_risk, 'debtToIncome':debtToIncome, 'gender':gender, 'location':location, 'propensityScore':propensityScore, 'account_age':account_age})

########################################
#Model Building
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

#"Importing" the Data; KPI must be in the first column
FakeData=FakeData
y=FakeData.iloc[:, 0]
X=FakeData.iloc[:, FakeData.columns!=FakeData.columns[0]]

#Labeling features as either categorical or numeric
nrows=X.shape[1]
is_categorical=[]
for i in range(0, nrows):
    if (len(set(X.iloc[:, i]))<=20):
        is_categorical.append(i)
is_numeric=[]
for i in range(0, nrows):
    if (len(set(X.iloc[:, i]))>20):
        is_numeric.append(i)

#Extracting out the numeric features
categorical_df=pd.DataFrame()
X_categorical=X.iloc[:, is_categorical]
X_categorical_saved=X_categorical
X_numeric=X.iloc[:, is_numeric]

#Encoding categorical features as binary
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
X_categorical=X_categorical.apply(le.fit_transform)
enc=preprocessing.OneHotEncoder()
enc.fit(X_categorical)
onehotlabels=enc.transform(X_categorical).toarray()
X_categorical=onehotlabels

nrows2=X_categorical.shape[1]

#Appropriately renaming the columns 
features=list(X_categorical_saved.columns)
len(features)
fvs=[]
for i in range(0, len(features)):
    fvs.append(list(set(X_categorical_saved.iloc[:, i])))
true_fvs=[]
set(X_categorical_saved.iloc[:,0])
fvs=list(list(fvs))

for i in range(0, len(features)):
    for j in range(0, len(fvs[i])):
        true_fvs.append(fvs[i][j])
col_names1=[]
col_names2=[]
for i in range(0, len(features)):
    for j in range(0, len(fvs[i])):
        col_names1.append(features[i]) ; col_names2.append(fvs[i][j])
col_names=[]
for i in range(0, len(set(col_names2))):
    col_names.append(col_names1[i]+":"+col_names2[i])

#Turning the categorical features into list form
X_df=pd.DataFrame()
holder=[]
for i in range(0, nrows2):
    holder=list(X_categorical[:, i])
    holder=pd.Series(holder, name=col_names[i])
    holder=pd.DataFrame(holder)
    #holder.reindex(axis=1)
    X_df=pd.concat([X_df, holder], axis=1)

X_new=pd.concat([X_df, X_numeric], axis=1)

#Running the Logistic Regression
logreg=LogisticRegression(C=1000000, fit_intercept=True)
logreg.fit(X_new, y).coef_

has_binary='Yes'

if (has_binary=='Yes'):
    #cONFIG_1
    #predictions=pd.Series(logreg.predict(X_new), name='Prediction')
    predictions=pd.Series(list(logreg.predict_proba(X_new)[:,1]), name='Prediction')
    X_new=pd.concat([X_new, predictions], axis=1)
    #Creating a unique cohort identifier for the existing data
    to_group=pd.concat([X_df, predictions], axis=1)

if (has_binary=='No'):
    NewData=X.iloc[0:1000, :]
    NewData=NewData
    #Everything until the following "***" is the same preprocessing that occurred with the original data

    #Labeling features as either categorical or numeric
    nrows=NewData.shape[1]
    is_categorical=[]
    for i in range(0, nrows):
        if (len(set(NewData.iloc[:, i]))<=20):
            is_categorical.append(i)
    is_numeric=[]
    for i in range(0, nrows):
        if (len(set(NewData.iloc[:, i]))>20):
            is_numeric.append(i)
    #Extracting out the numeric features
    categorical_df=pd.DataFrame()
    NewData_categorical=NewData.iloc[:, is_categorical]
    NewData_categorical_saved=NewData_categorical
    NewData_numeric=NewData.iloc[:, is_numeric]

    #Encoding categorical features as binary
    from sklearn import preprocessing
    le=preprocessing.LabelEncoder()
    NewData_categorical=NewData_categorical.apply(le.fit_transform)
    enc=preprocessing.OneHotEncoder()
    enc.fit(NewData_categorical)
    onehotlabels=enc.transform(NewData_categorical).toarray()
    NewData_categorical=onehotlabels

    nrows2=NewData_categorical.shape[1]

    #Appropriately renaming the columns 
    features=list(NewData_categorical_saved.columns)
    len(features)
    fvs=[]
    for i in range(0, len(features)):
        fvs.append(list(set(NewData_categorical_saved.iloc[:, i])))
    true_fvs=[]
    set(NewData_categorical_saved.iloc[:,0])
    fvs=list(list(fvs))
    fvs
    for i in range(0, len(features)):
        for j in range(0, len(fvs[i])):
            true_fvs.append(fvs[i][j])
    col_names1=[]
    col_names2=[]
    for i in range(0, len(features)):
        for j in range(0, len(fvs[i])):
            col_names1.append(features[i]) ; col_names2.append(fvs[i][j])
    col_names=[]
    for i in range(0, len(set(col_names2))):
        col_names.append(col_names1[i]+":"+col_names2[i])

    #Turning the categorical features into list form
    NewData_df=pd.DataFrame()
    holder=[]
    for i in range(0, nrows2):
        holder=list(NewData_categorical[:, i])
        holder=pd.Series(holder, name=col_names[i])
        holder=pd.DataFrame(holder)
        #holder.reindex(axis=1)
        NewData_df=pd.concat([NewData_df, holder], axis=1)

    NewData_new=pd.concat([NewData_df, NewData_numeric], axis=1)
    to_group=NewData_df

    #Predicting the probabilities of defaultment from the new data
    #CONFIG_1
    #predictions=pd.Series(list(logreg.predict(NewData_new)), name='Prediction')
    predictions=pd.Series(list(logreg.predict_proba(NewData_new)[:,1]), name='Prediction')

    to_group=pd.concat([to_group, predictions], axis=1)

#to_group

#Mapping each id to its Cohort
cohort=[]
for i in range(0, X_df.shape[0]):
    for j in range(0, X_df.shape[1]):
        if X_df.iloc[i, j]==1:
            cohort.append(X_df.columns[j])        

cohort_combined=[]
for i in range(0, int(len(cohort)/len(is_categorical))):
    cohort_combined.append('')
    for j in range(0, len(is_categorical)):
        cohort_combined[i]=cohort_combined[i]+cohort[int(len(is_categorical)*i+j)]+', '

cohort_combined=pd.Series(cohort_combined, name='Cohort')
to_group_with_cohorts=pd.concat([to_group, cohort_combined], axis=1)

#Grouping by Cohort
cohort_averages=to_group_with_cohorts.groupby('Cohort').mean()
cohort_counts=to_group_with_cohorts.groupby('Cohort').count()
averages_and_counts=pd.concat([cohort_averages['Prediction'], cohort_counts['Prediction']], axis=1)
averages_and_counts.columns=['Proportion That are Likely to Default', 'Count']
averages_and_counts

#Finding the cohorts who are the most influencable: that is, closest to 'default=0.5' border
#The higher the "influencable value", the more influencable the cohort is. Value ranges from 0 to 1
influencable_value=0.5
averages_and_counts['Influence Value']=1-abs(averages_and_counts['Proportion That are Likely to Default'] - influencable_value)
averages_and_counts=averages_and_counts.sort_values(by='Influence Value', ascending=False)
averages_and_counts.iloc[:, [1, 2]]
