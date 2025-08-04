# Employment Indicators: Rural

### Names: HABIYAREMYE Adolphe
### Id: 26751
### Course: Big Data Analysis

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

<img width="989" height="590" alt="6" src="https://github.com/user-attachments/assets/9be60c52-53e1-45c5-9b18-9da312451a64" />
<img width="1189" height="790" alt="5" src="https://github.com/user-attachments/assets/b9735cc2-975a-4c85-9e4f-d33873f91deb" />
<img width="988" height="590" alt="4" src="https://github.com/user-attachments/assets/ca731e4a-d9b5-443e-a6cd-b4e3a2ffa932" />
<img width="989" height="590" alt="3" src="https://github.com/user-attachments/assets/9d886c98-6f7c-46c3-80a1-4ba00a33d3dc" />
<img width="672" height="590" alt="2" src="https://github.com/user-attachments/assets/0cc6e096-27a8-430e-94aa-302cd941f5f6" />
<img width="1193" height="790" alt="1" src="https://github.com/user-attachments/assets/b626f360-dcb3-45f1-97d8-9150bf388082" />






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

# BACKEND AND FRONTED OF THIS APPLICATION
# Big Data Analytics Capstone Project (Employment Indicators: Rural)

A comprehensive web application for analyzing urban population data using Python (Flask) backend and React frontend, designed for AUCA's Introduction to Big Data Analytics course (INSY 8413).

## 🎯 Project Overview

**Sector:** Agriculture/Urban Development  
**Problem Statement:** Can we analyze and predict urban population growth patterns to understand global urbanization trends and support sustainable development planning?  
**Dataset:** FAOSTAT Urban Population Data or similar structured population datasets

## 🚀 Features

- **Data Upload & Processing**: Support for CSV and Excel files with intelligent parsing
- **Comprehensive Data Analysis**: Automated data quality assessment and statistical analysis
- **Interactive Data Cleaning**: Configurable cleaning options with real-time feedback
- **Dynamic Visualizations**: Time series, comparisons, and distribution charts
- **Machine Learning**: Population growth prediction using Random Forest and Linear Regression
- **Export Capabilities**: Download cleaned data and export visualizations

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn package manager

### Backend Setup

1. **Create project structure:**
```bash
mkdir urban-population-analyzer
cd urban-population-analyzer
mkdir backend frontend
```

2. **Set up Python backend:**
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Create the backend files:**
   - Copy all Python files (`app.py`, `data_processor.py`, `ml_models.py`)
   - Create `requirements.txt` with the specified dependencies
   - Create `uploads/` directory for file storage

4. **Run the Flask backend:**
```bash
python app.py
```
Backend will run on `http://localhost:5000`

### Frontend Setup

1. **Set up React frontend:**
```bash
cd ../frontend
npx create-react-app .
npm install axios recharts react-router-dom react-dropzone lucide-react @headlessui/react clsx
npm install -D tailwindcss autoprefixer postcss
```

2. **Configure Tailwind CSS:**
```bash
npx tailwindcss init -p
```

Update `tailwind.config.js`:
```javascript
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

3. **Replace default files:**
   - Copy all React component files to `src/components/`
   - Replace `src/App.js` with the provided App component
   - Replace `src/App.css` with the provided CSS
   - Update `package.json` with the provided configuration

4. **Run the React frontend:**
```bash
npm start
```
Frontend will run on `http://localhost:3000`

## 📁 Project Structure

```
urban-population-analyzer/
├── backend/
│   ├── app.py                 # Flask main application
│   ├── data_processor.py      # Data cleaning and analysis
│   ├── ml_models.py          # Machine learning models
│   ├── requirements.txt      # Python dependencies
│   └── uploads/             # File upload directory
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.js
│   │   │   ├── FileUpload.js
│   │   │   ├── DataAnalysis.js
│   │   │   ├── DataCleaning.js
│   │   │   ├── Visualization.js
│   │   │   ├── MachineLearning.js
│   │   │   └── ProjectInfo.js
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## 📊 Usage Guide

### 3. Data Format
Expected CSV columns:
- `Area` or `Country`: Geographic region
- `Year`: Time period
- `Value` or `Population`: Population numbers
- `Element` (optional): Data type identifier
- `Unit` (optional): Measurement unit

## 🔧 Configuration Options

### Data Cleaning
- **Missing Data**: Drop rows, fill with statistical measures, or keep as-is
- **Duplicates**: Automatic duplicate row removal
- **Outliers**: IQR-based outlier detection and removal
- **Data Types**: Automatic type conversion

### Machine Learning
- **Models**: Random Forest, Linear Regression
- **Prediction Period**: 3, 5, 10, or 15 years
- **Countries**: Select specific countries or auto-select top performers

## 📈 Export & Integration

### Power BI Integration
1. Export cleaned datasets from the cleaning module
2. Download visualization data for Power BI import
3. Use the generated insights for dashboard creation

# Screenshots of the project step by step
<img width="1919" height="1199" alt="Screenshot 2025-08-03 210313" src="https://github.com/user-attachments/assets/4f397cbe-59ae-4c23-bdbc-8f983e53cce3" /># FAOSTAT Data Analysis: A Comprehensive Data Science Workflow
![Uploading Screens<img width="1919" height="1199" alt="Screenshot 2025-08-03 210338" src="https://github.com/user-attachments/assets/f6358430-3664-409e-be0a-531b65b9d8fb" />
hot 2025-08-03 210313<img width="1919" height="1199" alt="Screenshot 2025-08-03 210414" src="https://github.com/user-attachments/assets/88e3c1bd-493e-45da-9252-4ebd6912e0bf" />
.png…]()
<img width="1503" height="474" alt="Screenshot 2025-08-03 210528" src="https://github.com/user-attachments/assets/189076a3-4ebc-4feb-9101-fea13b8f7c27" />
<img width="1919" height="1199" alt="Screenshot 2025-08-03 210705" src="https://github.com/user-attachments/assets/937d780b-44f7-4dd7-87dd-891bff15a1e9" />
<img width="1427" height="916" alt="Screenshot 2025-08-03 210742" src="https://github.com/user-attachments/assets/6ee8ae2c-ecdc-4104-b159-eeda28e1e018" />
<img width="1919" height="946" alt="Screenshot 2025-08-03 210841" src="https://github.com/user-attachments/assets/1f6b07cc-97be-449b-99ce-dedd828c0e52" />
<img width="1919" height="933" alt="Screenshot 2025-08-03 210951" src="https://github.com/user-attachments/assets/b5c417aa-7608-4a79-8939-5fc7f3a076a3" />
<img width="1919" height="935" alt="Screenshot 2025-08-03 211016" src="https://github.com/user-attachments/assets/d2eee59e-e2f9-45ae-80aa-2a9ec694150c" />
<img width="1914" height="901" alt="Screenshot 2025-08-03 211052" src="https://github.com/user-attachments/assets/30f25eab-fa52-4e36-b35f-2934aa6e9715" />
<img width="1919" height="917" alt="Screenshot 2025-08-03 211127" src="https://github.com/user-attachments/assets/70f5c94b-7edd-46b8-9c79-b5081d108a7c" />
<img width="1919" height="928" alt="Screenshot 2025-08-03 211155" src="https://github.com/user-attachments/assets/50db6ef8-ef8d-4cd9-bd94-5785b06ab970" />
<img width="1919" height="1199" alt="Screenshot 2025-08-03 211225" src="https://github.com/user-attachments/assets/c30d57a8-1510-4854-b182-15d6b0f31870" />


### GitHub Repository Structure
```
project-repository/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned datasets
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Source code
├── visualizations/          # Exported charts and graphs
├── reports/                 # Analysis reports
└── README.md               # Project documentation
```



### Environment Variables
Create `.env` file in backend directory:
```
FLASK_ENV=development
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
```

# POWER BI
# 📊 POWER BI DASHBOARD DESIGN GUIDE
## FAO Rural Employment Data Analysis Dashboard

**Document Version:** 1.0  
**Date:** August 2025  
**Purpose:** Complete visual reference for Power BI dashboard creation

---

# 📋 TABLE OF CONTENTS

1. [Dashboard Overview](#dashboard-overview)
2. [Complete Visual Layout](#complete-visual-layout)  
3. [Component Specifications](#component-specifications)
4. [Color Scheme & Branding](#color-scheme--branding)
5. [Chart Creation Steps](#chart-creation-steps)
6. [Interactive Elements](#interactive-elements)
7. [Mobile Layout](#mobile-layout)
8. [Quality Checklist](#quality-checklist)

---

# 📊 DASHBOARD OVERVIEW

## Key Statistics
- **Data Source:** FAO Rural Employment Dataset
- **Total Records:** 80,794 rows
- **Geographic Coverage:** 162 countries/regions
- **Time Period:** 2008-2020 (13 years)
- **Dashboard Components:** 11 visual elements + 4 interactive controls

## Business Objectives
1. **Geographic Analysis:** Show global distribution of rural employment
2. **Trend Analysis:** Display employment changes over time
3. **Demographic Insights:** Analyze gender and age patterns
4. **Comparative Analysis:** Identify top-performing countries
5. **Interactive Exploration:** Enable user-driven data discovery

---

# 🖼️ COMPLETE VISUAL LAYOUT

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                     🌾 FAO RURAL EMPLOYMENT DASHBOARD                          │
│              Global Analysis of Rural Employment Indicators (2008-2020)        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │     162     │    │   45.2M     │    │     13      │    │      6      │    │
│   │             │    │             │    │             │    │             │    │
│   │ COUNTRIES   │    │ EMPLOYMENT  │    │    YEARS    │    │   SOURCES   │    │
│   │  COVERED    │    │ (THOUSANDS) │    │   OF DATA   │    │   USED      │    │
│   │             │    │             │    │             │    │             │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                                                 │
├─────────────────────────────────────────┬───────────────────────────────────────┤
│                                         │                                       │
│   ┌─────────────────────────────────┐   │   ┌─────────────────────────────────┐ │
│   │                                 │   │   │        📈 EMPLOYMENT TRENDS     │ │
│   │         🌍 WORLD MAP            │   │   │                                 │ │
│   │                                 │   │   │   140k ●─────────────────────●  │ │
│   │      ●        ●      ●          │   │   │        │                       │ │
│   │   ●     ●   ●   ●  ●   ●        │   │   │   120k │     ●─────────●       │ │
│   │ ●   ●     ●       ●     ●       │   │   │        │    ╱               ╲   │ │
│   │   ●   ●     ●   ●     ●         │   │   │   100k │   ●                 ●  │ │
│   │     ●     ●   ●   ●             │   │   │        │  ╱     ── Total       │ │
│   │ ●     ●   ●     ●   ●           │   │   │    80k │ ●      ── Male        │ │
│   │   ●     ●   ●     ●             │   │   │        │╱       ── Female      │ │
│   │     ●     ●   ●     ●           │   │   │    60k ●                       │ │
│   │                                 │   │   │        2008  2012  2016  2020  │ │
│   │ (Bubble size = Employment Level)│   │   │                                 │ │
│   │                                 │   │   └─────────────────────────────────┘ │
│   └─────────────────────────────────┘   │                                       │
│                                         │                                       │
├─────────────────────────────────────────┤   ┌─────────────────────────────────┐ │
│                                         │   │     📊 YEAR-OVER-YEAR GROWTH   │ │
│   ┌─────────────────────────────────┐   │   │                                 │ │
│   │    📊 TOP 10 COUNTRIES          │   │   │   15% ┌─────────────────────────┐ │
│   │                                 │   │   │       │ ████████████████████████ │ │
│   │ China        ████████████████████│   │   │   10% │ ████████████████████     │ │
│   │ India        ███████████████████ │   │   │       │ ████████████████         │ │
│   │ Brazil       ████████████████    │   │   │    5% │ ████████████             │ │
│   │ Nigeria      ███████████████     │   │   │       │ ████████                 │ │
│   │ Indonesia    ██████████████      │   │   │    0% │ ████                     │ │
│   │ Pakistan     █████████████       │   │   │       │ ██                       │ │
│   │ Bangladesh   ████████████        │   │   │   -5% └─────────────────────────┘ │
│   │ Ethiopia     ███████████         │   │   │       2008 2010 2012 2014 2016  │ │
│   │ Vietnam      ██████████          │   │   │                                 │ │
│   │ Turkey       █████████           │   │   │       (Top 5 Countries)        │ │
│   │                                 │   │   │                                 │ │
│   │         0    50k   100k  150k   │   │   └─────────────────────────────────┘ │
│   │      Employment (Thousands)     │   │                                       │
│   │                                 │   │                                       │
│   └─────────────────────────────────┘   │                                       │
│                                         │                                       │
├─────────────────────────────────────────┴───────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────────┐ │
│   │      👥 GENDER DISTRIBUTION     │   │       📊 EMPLOYMENT BY AGE          │ │
│   │                                 │   │                                     │ │
│   │          ┌─────────────┐        │   │  Total (15+)     ████████████████████│ │
│   │        ██│████████ 45% │        │   │                                     │ │
│   │      ████│████████     │        │   │  15-24 Years     ████████████████    │ │
│   │    ██████│ 35%  ██████ │ 20%    │   │                                     │ │
│   │    ██████│    ████████ │        │   │  25-54 Years     ████████████████████│ │
│   │      ████│  ██████     │        │   │                                     │ │
│   │        ██│████         │        │   │  55-64 Years     ████████████        │ │
│   │          └─────────────┘        │   │                                     │ │
│   │                                 │   │  65+ Years       ████████            │ │
│   │   ■ Total  ■ Male  ■ Female     │   │                                     │ │
│   │                                 │   │                                     │ │
│   │   Total Employment Distribution │   │  ■ Total  ■ Male  ■ Female          │ │
│   │   by Gender Categories          │   │                                     │ │
│   │                                 │   │     0    50k   100k   150k   200k   │ │
│   └─────────────────────────────────┘   │      Employment (Thousands)         │ │
│                                         └─────────────────────────────────────┘ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   🎛️ INTERACTIVE CONTROLS                                                      │
│                                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │ COUNTRIES ▼ │  │ YEAR RANGE  │  │ GENDER      │  │ DATA SOURCE │           │
│   │             │  │             │  │             │  │             │           │
│   │ □ All       │  │ 2008 ═══════ │  │ ☑ Total     │  │ ▼ All       │           │
│   │ ☑ China     │  │       ══════ │  │ ☑ Male      │  │ □ Labor     │           │
│   │ ☑ India     │  │       ══ 2020│  │ ☑ Female    │  │   Force     │           │
│   │ □ Brazil    │  │             │  │             │  │ □ Census    │           │
│   │ □ Nigeria   │  │   [2010-2018]│  │             │  │ □ Survey    │           │
│   │ ⋮ (More)    │  │             │  │             │  │ □ Admin     │           │
│   │             │  │             │  │             │  │   Records   │           │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Dashboard Dimensions:** 1920px × 1080px (16:9 aspect ratio)  
**Grid System:** 12 columns × 8 rows  
**Spacing:** 10px margins, 15px gutters

---

# 🎨 COLOR SCHEME & BRANDING

## Primary Color Palette

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COLOR NAME     │  HEX CODE  │  RGB VALUES     │  USAGE                │
├─────────────────┼────────────┼─────────────────┼───────────────────────┤
│  Dark Blue      │  #1f4e79   │  31, 78, 121    │  Titles, Headers      │
│  Primary Blue   │  #2196f3   │  33, 150, 243   │  Male Data, Lines     │
│  Pink           │  #e91e63   │  233, 30, 99    │  Female Data          │
│  Green          │  #2d7d32   │  45, 125, 50    │  Positive Trends      │
│  Orange         │  #f57c00   │  245, 124, 0    │  Highlights           │
│  Gray           │  #757575   │  117, 117, 117  │  Total/Neutral        │
│  Light Gray     │  #f8f9fa   │  248, 249, 250  │  Background           │
│  White          │  #ffffff   │  255, 255, 255  │  Card Backgrounds     │
└─────────────────┴────────────┴─────────────────┴───────────────────────┘
```

## Typography Standards

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ELEMENT TYPE      │  FONT FAMILY  │  SIZE  │  WEIGHT │  COLOR          │
├────────────────────┼───────────────┼────────┼─────────┼─────────────────┤
│  Main Title        │  Segoe UI     │  28pt  │  Bold   │  #1f4e79        │
│  Subtitle          │  Segoe UI     │  16pt  │  Normal │  #666666        │
│  Chart Titles      │  Segoe UI     │  14pt  │  Bold   │  #1f4e79        │
│  Axis Labels       │  Segoe UI     │  10pt  │  Normal │  #424242        │
│  Data Labels       │  Segoe UI     │  9pt   │  Normal │  Auto           │
│  KPI Numbers       │  Segoe UI     │  24pt  │  Bold   │  Theme Color    │
│  KPI Labels        │  Segoe UI     │  10pt  │  Normal │  #666666        │
└────────────────────┴───────────────┴────────┴─────────┴─────────────────┘
```

---

# 📊 COMPONENT SPECIFICATIONS

## 1. KPI Cards (Top Row)

```
CARD 1: COUNTRIES COVERED
┌─────────────────────────────────┐
│ Position: Row 1, Columns 1-3    │
│ Size: 280px × 120px             │
│ ┌─────────────┐                 │
│ │     162     │ ← 24pt, Bold    │
│ │ COUNTRIES   │ ← 10pt, Normal  │
│ │  COVERED    │                 │
│ └─────────────┘                 │
│ Data Source: COUNT(DISTINCT Area)│
│ Color: #1f4e79                  │
└─────────────────────────────────┘

CARD 2: TOTAL EMPLOYMENT
┌─────────────────────────────────┐
│ Position: Row 1, Columns 4-6    │
│ Size: 280px × 120px             │
│ ┌─────────────┐                 │
│ │   45.2M     │ ← 24pt, Bold    │
│ │ EMPLOYMENT  │ ← 10pt, Normal  │
│ │(THOUSANDS)  │                 │
│ └─────────────┘                 │
│ Data Source: SUM(Value)         │
│ Color: #2d7d32                  │
└─────────────────────────────────┘

CARD 3: YEARS OF DATA
┌─────────────────────────────────┐
│ Position: Row 1, Columns 7-9    │
│ Size: 280px × 120px             │
│ ┌─────────────┐                 │
│ │     13      │ ← 24pt, Bold    │
│ │    YEARS    │ ← 10pt, Normal  │
│ │   OF DATA   │                 │
│ └─────────────┘                 │
│ Data Source: COUNT(DISTINCT Year)│
│ Color: #f57c00                  │
└─────────────────────────────────┘

CARD 4: DATA SOURCES
┌─────────────────────────────────┐
│ Position: Row 1, Columns 10-12  │
│ Size: 280px × 120px             │
│ ┌─────────────┐                 │
│ │      6      │ ← 24pt, Bold    │
│ │   SOURCES   │ ← 10pt, Normal  │
│ │    USED     │                 │
│ └─────────────┘                 │
│ Data Source: COUNT(DISTINCT Source)│
│ Color: #7b1fa2                  │
└─────────────────────────────────┘
```

## 2. World Map (Main Visual)

```
WORLD MAP SPECIFICATIONS
┌─────────────────────────────────────────────────────────┐
│ Position: Row 2-4, Columns 1-6                         │
│ Size: 580px × 360px                                    │
│ Visual Type: Map                                        │
│                                                         │
│ FIELD ASSIGNMENTS:                                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 📍 Location  │ Area (Country names)                 │ │
│ │ 📏 Size      │ Value (Employment numbers)           │ │
│ │ 💬 Tooltips  │ Area, Value, Indicator, Sex, Year    │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ FORMATTING:                                             │
│ • Map Style: Road                                       │
│ • Data Colors: Blue gradient (#e3f2fd to #0d47a1)      │
│ • Bubble Min Size: 5px                                  │
│ • Bubble Max Size: 40px                                 │
│ • Default Zoom: Auto-fit                                │
│ • Map Controls: Show zoom controls                      │
│                                                         │
│ INTERACTIONS:                                           │
│ • Cross-highlight other visuals                         │
│ • Drill-through to country detail page                  │
│ • Tooltip shows top 3 indicators per country           │
└─────────────────────────────────────────────────────────┘
```

## 3. Employment Trends Line Chart

```
LINE CHART SPECIFICATIONS
┌─────────────────────────────────────────────────────────┐
│ Position: Row 2-3, Columns 7-12                        │
│ Size: 580px × 240px                                    │
│ Visual Type: Line Chart                                │
│                                                         │
│ FIELD ASSIGNMENTS:                                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ➡️ X-Axis    │ Year                                 │ │
│ │ ⬆️ Y-Axis    │ Value (Employment)                   │ │
│ │ 🏷️ Legend    │ Sex (Total, Male, Female)            │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ FORMATTING:                                             │
│ • Line Colors: Total=#757575, Male=#2196f3, Female=#e91e63│
│ • Line Width: 3px                                       │
│ • Markers: Show, 6px diameter                          │
│ • Data Labels: Show values at peak points               │
│ • Y-Axis: Start at 0, format as thousands              │
│ • X-Axis: Show all years, rotate labels 0°             │
│ • Grid Lines: Horizontal only, light gray              │
│                                                         │
│ INTERACTIONS:                                           │
│ • Highlight corresponding data in other charts          │
│ • Tooltip shows % change from previous year            │
└─────────────────────────────────────────────────────────┘
```

## 4. Top 10 Countries Bar Chart

```
BAR CHART SPECIFICATIONS
┌─────────────────────────────────────────────────────────┐
│ Position: Row 4-5, Columns 1-6                         │
│ Size: 580px × 240px                                    │
│ Visual Type: Clustered Bar Chart                       │
│                                                         │
│ FIELD ASSIGNMENTS:                                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ⬆️ Y-Axis    │ Area (Country names)                 │ │
│ │ ➡️ X-Axis    │ Value (Employment)                   │ │
│ │ 🎯 Filters   │ Area (Top 10 by Sum of Value)        │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ FORMATTING:                                             │
│ • Bar Colors: Gradient from #4caf50 to #2e7d32         │
│ • Sort Order: Descending by Value                      │
│ • Data Labels: Show values, inside end                 │
│ • Y-Axis: Country names, no truncation                 │
│ • X-Axis: Employment (Thousands), start at 0           │
│ • Grid Lines: Vertical only, light gray               │
│                                                         │
│ FILTER SETUP:                                           │
│ • Filter Type: Top N                                    │
│ • Show Items: Top 10                                    │
│ • By Value: Sum of Value                               │
│ • Include Others: No                                    │
└─────────────────────────────────────────────────────────┘
```

## 5. Gender Distribution Pie Chart

```
PIE CHART SPECIFICATIONS
┌─────────────────────────────────────────────────────────┐
│ Position: Row 6, Columns 1-6                           │
│ Size: 580px × 200px                                    │
│ Visual Type: Pie Chart                                 │
│                                                         │
│ FIELD ASSIGNMENTS:                                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 🏷️ Legend    │ Sex                                  │ │
│ │ 📊 Values    │ Value (Employment)                   │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ FORMATTING:                                             │
│ • Slice Colors: Total=#757575, Male=#2196f3, Female=#e91e63│
│ • Data Labels: Show percentage and value               │
│ • Label Position: Outside                              │
│ • Legend Position: Right                               │
│ • Legend Font: 10pt                                    │
│ • Inner Radius: 20% (donut style)                     │
│                                                         │
│ INTERACTIONS:                                           │
│ • Click slice to filter other visuals                 │
│ • Tooltip shows detailed breakdown                     │
└─────────────────────────────────────────────────────────┘
```

## 6. Age Breakdown Stacked Bar Chart

```
STACKED BAR CHART SPECIFICATIONS
┌─────────────────────────────────────────────────────────┐
│ Position: Row 6, Columns 7-12                          │
│ Size: 580px × 200px                                    │
│ Visual Type: Stacked Bar Chart                         │
│                                                         │
│ FIELD ASSIGNMENTS:                                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ⬆️ Y-Axis    │ Indicator (Age groups)               │ │
│ │ ➡️ X-Axis    │ Value (Employment)                   │ │
│ │ 🏷️ Legend    │ Sex                                  │ │
│ │ 🎯 Filters   │ Indicator (Contains "Age")           │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ FORMATTING:                                             │
│ • Stack Colors: Male=#2196f3, Female=#e91e63           │
│ • Sort Order: By total value, descending               │
│ • Data Labels: Show for segments >5%                   │
│ • Y-Axis: Age group labels, wrap text                  │
│ • X-Axis: Employment (Thousands)                       │
│ • Legend: Bottom, horizontal                           │
│                                                         │
│ FILTER SETUP:                                           │
│ • Filter Type: Contains                                 │
│ • Search Term: "Age"                                    │
│ • Include: All age-related indicators                  │
└─────────────────────────────────────────────────────────┘
```

---

# 🎛️ INTERACTIVE ELEMENTS

## Slicer Configurations

```
SLICER 1: COUNTRY SELECTION
┌─────────────────────────────────────────────────────────┐
│ Position: Row 7, Columns 1-3                           │
│ Size: 280px × 150px                                    │
│ Style: List                                             │
│                                                         │
│ SETTINGS:                                               │
│ • Field: Area                                          │
│ • Selection Type: Multi-select                         │
│ • Search Box: Enabled                                  │
│ • Select All: Enabled                                  │
│ • Max Items Visible: 8                                 │
│ • Scroll Bar: Auto                                     │
│                                                         │
│ FORMATTING:                                             │
│ • Header: "SELECT COUNTRIES"                           │
│ • Header Color: #1f4e79                               │
│ • Items Font: 10pt                                     │
│ • Selected Color: #e3f2fd                             │
│ • Hover Color: #f5f5f5                                │
└─────────────────────────────────────────────────────────┘

SLICER 2: YEAR RANGE
┌─────────────────────────────────────────────────────────┐
│ Position: Row 7, Columns 4-6                           │
│ Size: 280px × 150px                                    │
│ Style: Between                                          │
│                                                         │
│ SETTINGS:                                               │
│ • Field: Year                                          │
│ • Range Type: Between                                  │
│ • Show Range: Enabled                                  │
│ • Step Size: 1                                         │
│                                                         │
│ FORMATTING:                                             │
│ • Header: "SELECT YEAR RANGE"                          │
│ • Slider Color: #2196f3                               │
│ • Handle Color: #1f4e79                               │
│ • Range Display: Show min-max                          │
└─────────────────────────────────────────────────────────┘

SLICER 3: GENDER SELECTION
┌─────────────────────────────────────────────────────────┐
│ Position: Row 7, Columns 7-9                           │
│ Size: 280px × 150px                                    │
│ Style: Button                                          │
│                                                         │
│ SETTINGS:                                               │
│ • Field: Sex                                           │
│ • Selection Type: Multi-select                         │
│ • Orientation: Vertical                                │
│                                                         │
│ FORMATTING:                                             │
│ • Header: "SELECT GENDER"                              │
│ • Button Colors: Unselected=#f5f5f5, Selected=#2196f3 │
│ • Text Color: #1f4e79                                 │
│ • Button Border: 1px solid #ddd                       │
└─────────────────────────────────────────────────────────┘

SLICER 4: DATA SOURCE
┌─────────────────────────────────────────────────────────┐
│ Position: Row 7, Columns 10-12                         │
│ Size: 280px × 150px                                    │
│ Style: Dropdown                                         │
│                                                         │
│ SETTINGS:                                               │
│ • Field: Source                                        │
│ • Selection Type: Multi-select                         │
│ • Default: All selected                                │
│                                                         │
│ FORMATTING:                                             │
│ • Header: "SELECT DATA SOURCE"                         │
│ • Dropdown Arrow: #1f4e79                             │
│ • Selected Items: Show count                           │
│ • Placeholder: "All Sources Selected"                  │
└─────────────────────────────────────────────────────────┘
```

---

# 📱 MOBILE LAYOUT

## Mobile Optimization (Portrait 375px × 667px)

```
┌─────────────────────────┐
│    🌾 FAO EMPLOYMENT    │ ← Condensed title
│         DASHBOARD       │
├─────────────────────────┤
│ ┌─────┐ ┌─────┐        │ ← 2×2 KPI grid
│ │ 162 │ │45.2M│        │
│ │CNTRY│ │EMPL │        │
│ └─────┘ └─────┘        │
│ ┌─────┐ ┌─────┐        │
│ │  13 │ │  6  │        │
│ │YEARS│ │SRCS │        │
│ └─────┘ └─────┘        │
├─────────────────────────┤
│     🌍 WORLD MAP        │ ← Full width, reduced height
│                         │
│   (Touch to explore)    │
│                         │
├─────────────────────────┤
│   📈 EMPLOYMENT TRENDS  │ ← Simplified line chart
│                         │
│ ●────●────●────●        │
│ 2008 2012 2016 2020    │
│                         │
├─────────────────────────┤
│  📊 TOP 5 COUNTRIES     │ ← Reduced from top 10
│                         │
│ China     ████████████  │
│ India     ███████████   │
│ Brazil    ████████      │
│ Nigeria   ███████       │
│ Indonesia █████         │
│                         │
├─────────────────────────┤
│ 👥 GENDER │ 📊 AGE      │ ← Side by side, compact
│           │             │
│  ●45%     │ 15+ ████    │
│ ●35% ●20% │ 24+ ███     │
│           │ 55+ ██      │
├─────────────────────────┤
│ 🎛️ QUICK FILTERS        │ ← Stacked filters
│ [All Countries    ▼]    │

```
<img width="1920" height="1200" alt="Screenshot 2025-08-04 133736" src="https://github.com/user-attachments/assets/ed9ad274-3656-45cb-b0fa-01d222d7c0db" />

<img width="1920" height="1200" alt="Screenshot 2025-08-04 133842" src="https://github.com/user-attachments/assets/44089685-ddc5-446e-b665-797c4a4ca022" />

<img width="1920" height="1200" alt="Screenshot 2025-08-04 133819" src="https://github.com/user-attachments/assets/82394592-7a8a-484f-9fd4-f7fff1a0d078" />
<img width="1920" height="1200" alt="Screenshot 2025-08-04 133911" src="https://github.com/user-attachments/assets/d7d77430-f35f-4fdb-a067-e57f54efebd3" />


## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Data Engineering**: ETL processes and data quality assessment
- **Statistical Analysis**: Descriptive statistics and trend analysis
- **Machine Learning**: Supervised learning and model evaluation
- **Data Visualization**: Interactive charts and dashboard design
- **Web Development**: Full-stack application development
- **Project Management**: Structured development and documentation

## Conclusion

This employment indicators domain focuses on indicators related to employment in agrifood systems and rural areas. The update is performed yearly, using data from the International Labour Organization (ILO) database that contains a rich set of indicators from a wide range of topics related to labour statistics. The indicators published in FAOSTAT are derived from the labour force statistics (LFS) and rural and urban labour markets (RURURB) databases of the ILOSTAT database.

**Technical Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib, SHAP
**Dataset**: FAOSTAT Agricultural and Food Security Indicators
**Analysis Type**: Descriptive, Predictive, and Prescriptive Analytics
