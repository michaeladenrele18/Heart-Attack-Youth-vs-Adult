import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


df = pd.read_csv("heart_attack_youth_vs_adult.csv")
"""
### 1. Plot for Sleep Hours ###
# Group by Sleep Hours and Age Group
sleep_hours_agg = df.groupby(['Sleep_Hours', 'Age_Group'])['Heart_Attack'].agg(['count', 'sum', 'mean']).reset_index()

# Plot Sleep Hours by Age Group
plt.figure(figsize=(12, 6))  
sns.lineplot(data=sleep_hours_agg, x='Sleep_Hours', y='mean', hue='Age_Group', marker='o')
plt.title('Heart Attack Rate by Sleep Hours (Youth vs Adults)', fontsize=16)
plt.xlabel('Sleep Hours', fontsize=12)
plt.ylabel('Heart Attack Rate (%)', fontsize=12)
plt.xticks(rotation=45, fontsize=10) 
plt.legend(title='Age Group', fontsize=10)
plt.show()

### 2. Plot for Smoking Status ###
# Group by Smoking Status and Age Group
smoking_status_agg = df.groupby(['Smoking_Status', 'Age_Group'])['Heart_Attack'].agg(['count', 'sum', 'mean']).reset_index()

# Plot Smoking Status by Age Group
plt.figure(figsize=(12, 6))
sns.barplot(data=smoking_status_agg, x='Smoking_Status', y='mean', hue='Age_Group', dodge=True)
plt.title('Heart Attack Rate by Smoking Status (Youth vs Adults)', fontsize=16)
plt.xlabel('Smoking Status', fontsize=12)
plt.ylabel('Heart Attack Rate (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.legend(title='Age Group', fontsize=10)
plt.show()

### 3. Plot for Diet Quality ###
# Group by Diet Quality and Age Group
diet_quality_agg = df.groupby(['Diet_Quality', 'Age_Group'])['Heart_Attack'].agg(['count', 'sum', 'mean']).reset_index()

# Plot Diet Quality by Age Group
plt.figure(figsize=(12, 6))
sns.barplot(data=diet_quality_agg, x='Diet_Quality', y='mean', hue='Age_Group', dodge=True)
plt.title('Heart Attack Rate by Diet Quality (Youth vs Adults)', fontsize=16)
plt.xlabel('Diet Quality', fontsize=12)
plt.ylabel('Heart Attack Rate (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.legend(title='Age Group', fontsize=10)
plt.show()

### 4. Plot for Alcohol Consumption ###
# Group by Alcohol Consumption and Age Group
alcohol_consumption_agg = df.groupby(['Alcohol_Consumption', 'Age_Group'])['Heart_Attack'].agg(['count', 'sum', 'mean']).reset_index()

# Plot Alcohol Consumption by Age Group
plt.figure(figsize=(12, 6))
sns.barplot(data=alcohol_consumption_agg, x='Alcohol_Consumption', y='mean', hue='Age_Group', dodge=True)
plt.title('Heart Attack Rate by Alcohol Consumption (Youth vs Adults)', fontsize=16)
plt.xlabel('Alcohol Consumption', fontsize=12)
plt.ylabel('Heart Attack Rate (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.legend(title='Age Group', fontsize=10)
plt.show()

### 5. Plot for Physical Activity ###
# Group by Physical Activity (Binned) and Age Group
physical_activity_agg = df.groupby(['Physical_Activity_Binned', 'Age_Group'])['Heart_Attack'].agg(['count', 'sum', 'mean']).reset_index()

# Plot Physical Activity by Age Group
plt.figure(figsize=(12, 6))
sns.barplot(data=physical_activity_agg, x='Physical_Activity_Binned', y='mean', hue='Age_Group', dodge=True)
plt.title('Heart Attack Rate by Physical Activity (Youth vs Adults)', fontsize=16)
plt.xlabel('Minutes of Physical Activity Per Week (Binned)', fontsize=12)
plt.ylabel('Heart Attack Rate (%)', fontsize=12)
plt.xticks(rotation=45, fontsize=10) 
plt.legend(title='Age Group', fontsize=10)
plt.show()

"""
#Chi Square Test For Categorical Variables
contingency_table = pd.crosstab(df['Age_Group'], df['Smoking_Status'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"P-value: {p}")


#T-Test For Numerical Variables
youth_data = df[df['Age_Group'] == 'Youth']['Heart_Attack']
adult_data = df[df['Age_Group'] == 'Adult']['Heart_Attack']
t_stat, p_value = ttest_ind(youth_data, adult_data)
print(f"T-Test P-value: {p_value}")
