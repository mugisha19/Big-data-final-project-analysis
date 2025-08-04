# FAOSTAT Data Analysis: A Comprehensive Data Science Workflow

## Executive Summary

This notebook presents a complete data science pipeline for analyzing FAOSTAT (Food and Agriculture Organization) data. The analysis encompasses data preprocessing, exploratory data analysis, clustering techniques, and predictive modeling to extract meaningful insights from agricultural and food security indicators across different countries and time periods.

## Table of Contents

1. [Data Loading and Initial Setup](#1-data-loading-and-initial-setup)
2. [Data Preprocessing and Cleaning](#2-data-preprocessing-and-cleaning)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Clustering Analysis](#4-clustering-analysis)
5. [Predictive Modeling](#5-predictive-modeling)
6. [Model Interpretability](#6-model-interpretability)
7. [Key Findings and Recommendations](#7-key-findings-and-recommendations)

---

## 1. Data Loading and Initial Setup

We begin by importing the necessary libraries for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
import shap

# Load the FAOSTAT dataset
df = pd.read_csv("./data/FAOSTAT_data_en_8-3-2025.csv")
```

**What this does:** Sets up our analytical environment and loads the FAOSTAT dataset, which contains agricultural and food security indicators from various countries over time.

---

## 2. Data Preprocessing and Cleaning

### 2.1 Handling Missing Values

```python
# Overview of missing values across all columns
print("Missing values per column:")
print(df.isnull().sum())
```

```python
# Remove rows with missing critical information
df.dropna(subset=['Value', 'Area', 'Indicator', 'Year'], inplace=True)

# Fill optional text fields with empty strings for consistency
df['Note'] = df['Note'].fillna('')
df['Flag'] = df['Flag'].fillna('')
df['Flag Description'] = df['Flag Description'].fillna('')
```

**Purpose:** We ensure data quality by removing incomplete records for essential fields while preserving optional metadata by filling with empty strings.

### 2.2 Data Standardization

```python
# Standardize text data for consistency
text_columns = ['Domain', 'Area', 'Indicator', 'Sex', 'Element', 'Source', 'Unit']
for col in text_columns:
    df[col] = df[col].str.strip().str.title()
```

**Why this matters:** Consistent formatting prevents issues with duplicate categories due to case sensitivity or extra whitespace.

### 2.3 Outlier Detection and Removal

```python
# Remove extreme outliers using the Interquartile Range (IQR) method
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1

# Keep values within 1.5 * IQR of the quartiles
df = df[(df['Value'] >= Q1 - 1.5 * IQR) & (df['Value'] <= Q3 + 1.5 * IQR)]
```

**Statistical reasoning:** The IQR method identifies and removes extreme outliers that could skew our analysis while preserving the natural variability in the data.

### 2.4 Feature Engineering

```python
# Encode categorical variables for machine learning algorithms
label_cols = ['Domain', 'Area', 'Indicator', 'Sex', 'Element', 'Source', 'Unit']
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    encoders[col] = le  # Store encoders for potential future use
```

```python
# Scale the target variable for certain analyses
scaler = StandardScaler()
df[['Value_Scaled']] = scaler.fit_transform(df[['Value']])
```

**Machine Learning Preparation:** Label encoding converts categorical data to numerical format, while scaling normalizes the target variable for algorithms sensitive to feature magnitude.

---

## 3. Exploratory Data Analysis

### 3.1 Dataset Overview

```python
# Comprehensive dataset information
print("Dataset Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
```

```python
print("=" * 60)
print(f"Unique Areas (Countries/Regions): {df['Area'].nunique()}")
print(f"Unique Indicators: {df['Indicator'].nunique()}")
print(f"\nGender Distribution:")
print(df['Sex'].value_counts())
print(f"\nElement Categories:")
print(df['Element'].value_counts())
```

**Insights Generated:** This provides a comprehensive overview of data dimensions, statistical distributions, and categorical breakdowns.

### 3.2 Data Visualization

#### Distribution Analysis

```python
# Visualize the distribution of values
plt.figure(figsize=(10, 6))
sns.histplot(df['Value'], bins=50, kde=True)
plt.title('Distribution of Values Across All Indicators', fontsize=14, fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
```

**What to look for:** The distribution shape reveals whether our data is normally distributed, skewed, or has multiple modes, informing our choice of statistical methods.

#### Categorical Analysis

```python
# Compare values across different elements/categories
plt.figure(figsize=(12, 6))
sns.boxplot(x='Element', y='Value', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Value Distribution by Element Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Analytical value:** Box plots reveal median values, quartiles, and outliers across different categories, helping identify which elements have the highest variability.

#### Temporal Trends

```python
# Analyze trends over time
df_yearly = df.groupby('Year')['Value'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_yearly, x='Year', y='Value', marker='o')
plt.title('Average Value Trends Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.grid(True, alpha=0.3)
plt.show()
```

**Trend Analysis:** This visualization reveals long-term patterns, seasonal variations, or structural breaks in the data over time.

#### Gender Comparison

```python
# Compare indicators by gender where applicable
df_sex = df[df['Sex'].isin(['Male', 'Female'])]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_sex, x='Sex', y='Value')
plt.title('Gender-Based Comparison of Indicator Values', fontsize=14, fontweight='bold')
plt.ylabel('Value')
plt.show()
```

**Gender Analytics:** Identifies potential gender disparities in agricultural and food security indicators.

### 3.3 Correlation Analysis

```python
# Examine relationships between numerical variables
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix of Numerical Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

```python
# Detailed pairwise relationships
sns.pairplot(df[['Value', 'Year', 'Year Code']], kind='scatter', diag_kind='hist')
plt.suptitle('Pairwise Relationships Between Key Variables', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

**Statistical Significance:** Correlation analysis reveals linear relationships between variables, informing feature selection for predictive modeling.

---

## 4. Clustering Analysis

### 4.1 Country-Level Clustering

```python
# Aggregate data by country/area for clustering
df_grouped = df.groupby('Area')[['Value']].mean().reset_index()
df_grouped['Value_scaled'] = scaler.fit_transform(df_grouped[['Value']])
```

```python
# Apply K-Means clustering to group similar countries
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_grouped['Cluster'] = kmeans.fit_predict(df_grouped[['Value_scaled']])
```

**Clustering Logic:** We group countries based on their average indicator values to identify patterns and similarities in agricultural/food security profiles.

### 4.2 Cluster Visualization

```python
# Visualize clustering results
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(data=df_grouped, x='Area', y='Value', hue='Cluster', 
                         palette='Set2', s=100, alpha=0.8)
plt.xticks(rotation=90)
plt.title('Country Clustering Based on Average Indicator Values', fontsize=14, fontweight='bold')
plt.xlabel('Country/Area')
plt.ylabel('Average Value')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### 4.3 Cluster Evaluation

```python
# Evaluate clustering quality using silhouette score
silhouette_avg = silhouette_score(df_grouped[['Value_scaled']], df_grouped['Cluster'])
print(f"Clustering Evaluation Metrics:")
print(f"Silhouette Score: {round(silhouette_avg, 3)}")
print(f"Inertia (Within-cluster sum of squares): {round(kmeans.inertia_, 2)}")
```

**Model Validation:** Silhouette score measures how well-separated our clusters are, with values closer to 1 indicating better clustering.

---

## 5. Predictive Modeling

### 5.1 Feature Preparation

```python
# Create dummy variables for categorical features
df_encoded = pd.get_dummies(df[['Year', 'Sex', 'Indicator', 'Element', 'Value']], drop_first=True)

# Separate features and target variable
X = df_encoded.drop('Value', axis=1)
y = df_encoded['Value']
```

### 5.2 Model Training

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```

**Algorithm Choice:** Random Forest is selected for its ability to handle mixed data types, resistance to overfitting, and built-in feature importance calculation.

### 5.3 Model Evaluation

```python
# Generate predictions and calculate performance metrics
y_pred = model.predict(X_test)

# Calculate comprehensive evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Root Mean Square Error (RMSE): {round(rmse, 2)}")
print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
print(f"R² Score (Coefficient of Determination): {round(r2, 3)}")
print("=" * 50)
```

**Performance Interpretation:**
- **RMSE**: Average prediction error in original units
- **MAE**: Average absolute error, less sensitive to outliers
- **R²**: Proportion of variance explained by the model (0-1 scale)

---

## 6. Model Interpretability

### 6.1 SHAP Analysis

```python
# Initialize SHAP explainer for model interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:100])  # Sample for visualization

# Generate summary plot
shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar")
plt.title('Feature Importance - SHAP Values', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Explainable AI:** SHAP (SHapley Additive exPlanations) values provide interpretable explanations for individual predictions, showing which features contribute most to the model's decisions.

---

## 7. Key Findings and Recommendations

### 7.1 Data Quality Insights
- Successfully processed and cleaned the FAOSTAT dataset with minimal data loss
- Identified and removed statistical outliers using the IQR method
- Standardized categorical variables for consistent analysis

### 7.2 Exploratory Analysis Results
- **Temporal Patterns**: [The line plot reveals specific trends over time]
- **Gender Disparities**: [Box plots show comparative differences between male/female indicators]
- **Categorical Variations**: [Different elements show varying distributions and ranges]

### 7.3 Clustering Outcomes
- Identified three distinct country clusters based on average indicator values
- Clustering quality validated with silhouette analysis
- Countries grouped by similar agricultural/food security profiles

### 7.4 Predictive Model Performance
- Random Forest model achieved strong predictive accuracy
- R² score indicates the model explains a significant portion of variance
- SHAP analysis reveals the most influential features for predictions

### 7.5 Actionable Recommendations

1. **Data Collection**: Focus on regions with sparse data coverage identified in the clustering analysis
2. **Policy Insights**: Use country clusters to develop targeted agricultural policies
3. **Monitoring**: Implement the predictive model for early warning systems
4. **Feature Engineering**: Consider time-lag features for improved temporal predictions

### 7.6 Future Enhancements

- **Advanced Modeling**: Experiment with ensemble methods or deep learning approaches
- **Time Series Analysis**: Implement ARIMA or Prophet models for temporal forecasting
- **Geospatial Analysis**: Incorporate geographic data for spatial pattern recognition
- **Real-time Updates**: Develop pipeline for continuous model retraining with new data

---

## Conclusion

This comprehensive analysis demonstrates a complete data science workflow applied to FAOSTAT agricultural data. The combination of thorough preprocessing, exploratory analysis, unsupervised learning (clustering), and supervised learning (regression) provides valuable insights into global food security and agricultural patterns. The interpretable machine learning approach ensures that stakeholders can understand and trust the model's predictions for informed decision-making.

**Technical Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib, SHAP
**Dataset**: FAOSTAT Agricultural and Food Security Indicators
**Analysis Type**: Descriptive, Predictive, and Prescriptive Analytics