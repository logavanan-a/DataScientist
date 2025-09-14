#!/usr/bin/env python
# coding: utf-8

# ### LOGISTIC REGRESSION

# #### 1. Data Exploration:

# In[176]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# ##### a. Load the dataset and perform exploratory data analysis (EDA).

# In[177]:


df = pd.read_csv("diabetes.csv")


# In[178]:


df.shape


# In[179]:


df.head()


# In[180]:


df.info()


# In[181]:


df.isna().sum()


# In[182]:


df.describe()


# ##### b. Examine the features, their types, and summary statistics.

# ##### c. Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.

# In[183]:


sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


# In[184]:


sns.boxplot(x='Outcome', y='Age', data=df[['Age','Outcome']])
plt.title("Age vs Outcome")
plt.show()


# In[185]:


sns.countplot(x='Outcome', hue='Glucose', data=df[['Glucose','Outcome']])
plt.title("Glucose vs Diabetes Outcome")
plt.show()


# ##### Analyze any patterns or correlations observed in the data.

# In[186]:


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# #### 2. Data Preprocessing:

# In[ ]:





# ##### a. Handle missing values (e.g., imputation).

# In[187]:


df.isna().sum()


# In[188]:


df.duplicated().sum()


# In[189]:


X=df[df.columns[:-1]]
X


# In[190]:


df.columns[-1] == 1
df


# In[191]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[192]:


le = LabelEncoder()
le.fit(y_train)


# In[193]:


le.classes_


# In[194]:


le.transform(y_train)


# #### 3. Model Building:

# ##### a. Build a logistic regression model using appropriate libraries (e.g., scikit-learn).

# ##### b. Train the model using the training data.

# In[195]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# #### 4. Model Evaluation:
# a. Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
# 

# In[196]:


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]


# In[197]:


accuracy_score(y_test, y_pred)


# In[198]:


precision_score(y_test, y_pred)


# In[199]:


recall_score(y_test, y_pred)


# In[200]:


f1_score(y_test, y_pred)


# In[201]:


roc_auc_score(y_test, y_prob)


# ##### Visualize the ROC curve.`

# In[202]:


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC curve (AUC = %.2f)" % roc_auc_score(y_test, y_prob))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()


# #### 5. Interpretation:
# a. Interpret the coefficients of the logistic regression model.
# 

# ###### b. Discuss the significance of features in predicting the target variable (survival probability in this case).

# In[203]:


coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})
print(coeff_df.sort_values(by="Coefficient", ascending=False))


# #### 6. Deployment with Streamlit:
# In this task, you will deploy your logistic regression model using Streamlit. The deployment can be done locally or online via Streamlit Share. Your task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions
# 

# In[204]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[151]:


import streamlit as st
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[205]:


joblib.dump((scaler, model), "logistic_model.pkl")


# In[206]:


import streamlit as st
import numpy as np
import joblib

# Load scaler and model
scaler, model = joblib.load("logistic_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter the values below to check the probability of diabetes.")

# Collect user inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Prediction: **{'Diabetic' if prediction == 1 else 'Not Diabetic'}**")
    st.write(f"Probability of Diabetes: {probability:.2f}")


# In[ ]:





# In[ ]:




