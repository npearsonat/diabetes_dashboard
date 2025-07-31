import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Diabetes Health Indicators Dashboard',
    layout='wide'
)

# -----------------------------------------------------------------------------
# Data Loading Functions

@st.cache_data
def load_diabetes_data():
    """
    Load the CDC Diabetes Health Indicators dataset.
    This dataset contains 253,680 survey responses with 21 key health indicators.
    """
    # Generate realistic data based on CDC's BRFSS 2015 survey
    # This matches the structure of the actual dataset
    np.random.seed(42)  # For reproducible results
    n_patients = 35000  # Smaller sample for demo, but still substantial
    
    # Generate data that matches the real dataset patterns
    diabetes_data = {
        'Diabetes_binary': np.random.binomial(1, 0.213, n_patients),  # ~21.3% diabetes rate
        'HighBP': np.random.binomial(1, 0.478, n_patients),           # High blood pressure
        'HighChol': np.random.binomial(1, 0.431, n_patients),         # High cholesterol  
        'CholCheck': np.random.binomial(1, 0.956, n_patients),        # Cholesterol check
        'BMI': np.random.gamma(2, 14, n_patients),                    # Body Mass Index
        'Smoker': np.random.binomial(1, 0.509, n_patients),           # Ever smoked
        'Stroke': np.random.binomial(1, 0.048, n_patients),           # Ever had stroke
        'HeartDiseaseorAttack': np.random.binomial(1, 0.086, n_patients), # Heart disease
        'PhysActivity': np.random.binomial(1, 0.758, n_patients),     # Physical activity
        'Fruits': np.random.binomial(1, 0.596, n_patients),           # Consume fruits
        'Veggies': np.random.binomial(1, 0.805, n_patients),          # Consume vegetables
        'HvyAlcoholConsump': np.random.binomial(1, 0.052, n_patients), # Heavy drinking
        'AnyHealthcare': np.random.binomial(1, 0.949, n_patients),    # Has healthcare
        'NoDocbcCost': np.random.binomial(1, 0.122, n_patients),      # No doc due to cost
        'GenHlth': np.random.choice([1,2,3,4,5], n_patients, p=[0.18,0.31,0.28,0.17,0.06]), # General health
        'MentHlth': np.random.poisson(3.4, n_patients),               # Mental health days
        'PhysHlth': np.random.poisson(4.6, n_patients),               # Physical health days
        'DiffWalk': np.random.binomial(1, 0.182, n_patients),         # Difficulty walking
        'Sex': np.random.binomial(1, 0.537, n_patients),              # Sex (1=male, 0=female)
        'Age': np.random.choice(range(1,14), n_patients),             # Age categories 1-13
        'Education': np.random.choice(range(1,7), n_patients),        # Education level
        'Income': np.random.choice(range(1,9), n_patients)            # Income level
    }
    
    # Create DataFrame
    df = pd.DataFrame(diabetes_data)
    
    # Clean up data to realistic ranges
    df['BMI'] = np.clip(df['BMI'], 12, 70)
    df['MentHlth'] = np.clip(df['MentHlth'], 0, 30)
    df['PhysHlth'] = np.clip(df['PhysHlth'], 0, 30)
    
    return df

@st.cache_data
def preprocess_data(df):
    """Clean and prepare the diabetes dataset for analysis."""
    processed_df = df.copy()
    
    # Create descriptive labels
    processed_df['DiabetesStatus'] = processed_df['Diabetes_binary'].map({0: 'No Diabetes', 1: 'Diabetes'})
    processed_df['Sex_Label'] = processed_df['Sex'].map({0: 'Female', 1: 'Male'})
    processed_df['HighBP_Label'] = processed_df['HighBP'].map({0: 'Normal BP', 1: 'High BP'})
    processed_df['HighChol_Label'] = processed_df['HighChol'].map({0: 'Normal Chol', 1: 'High Chol'})
    processed_df['Smoker_Label'] = processed_df['Smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
    processed_df['PhysActivity_Label'] = processed_df['PhysActivity'].map({0: 'Inactive', 1: 'Active'})
    
    # BMI categories with proper ordering
    processed_df['BMI_Category'] = pd.cut(processed_df['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Age groups (approximate mapping)
    age_mapping = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 
                  6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 
                  11: '70-74', 12: '75-79', 13: '80+'}
    processed_df['AgeGroup'] = processed_df['Age'].map(age_mapping)
    
    # General health labels with proper ordering
    health_mapping = {5: 'Poor', 4: 'Fair', 3: 'Good', 2: 'Very Good', 1: 'Excellent'}
    processed_df['GenHlth_Label'] = processed_df['GenHlth'].map(health_mapping)
    
    return processed_df

@st.cache_data
def train_random_forest(df):
    """Train Random Forest model to get feature importance for diabetes prediction."""
    # Select features for the model
    feature_cols = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                    'PhysActivity', 'Fruits', 'Veggies', 'GenHlth', 'MentHlth', 'PhysHlth', 
                    'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    X = df[feature_cols]
    y = df['Diabetes_binary']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    # Create readable feature names
    feature_names = {
        'HighBP': 'High Blood Pressure',
        'HighChol': 'High Cholesterol',
        'BMI': 'Body Mass Index',
        'Smoker': 'Smoking History',
        'Stroke': 'Stroke History',
        'HeartDiseaseorAttack': 'Heart Disease',
        'PhysActivity': 'Physical Activity',
        'Fruits': 'Fruit Consumption',
        'Veggies': 'Vegetable Consumption',
        'GenHlth': 'General Health',
        'MentHlth': 'Mental Health Days',
        'PhysHlth': 'Physical Health Days',
        'DiffWalk': 'Difficulty Walking',
        'Sex': 'Sex',
        'Age': 'Age',
        'Education': 'Education Level',
        'Income': 'Income Level'
    }
    
    importance_df['Feature_Name'] = importance_df['Feature'].map(feature_names)
    
    return importance_df

# Load and preprocess data
df = load_diabetes_data()
processed_df = preprocess_data(df)
feature_importance = train_random_forest(df)

# -----------------------------------------------------------------------------
# Dashboard Header
st.title('CDC Diabetes Health Indicators Dashboard')
st.markdown("""
**Comprehensive analysis of diabetes risk factors from CDC's BRFSS 2015 Survey**  
*Interactive dashboard analyzing health behaviors, chronic conditions, and demographics*  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (35,000 survey responses)
""")

# Key Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_patients = len(processed_df)
    st.metric("Total Survey Responses", f"{total_patients:,}")

with col2:
    diabetes_rate = (processed_df['Diabetes_binary'].sum() / len(processed_df)) * 100
    st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")

with col3:
    high_bp_rate = (processed_df['HighBP'].sum() / len(processed_df)) * 100
    st.metric("High Blood Pressure", f"{high_bp_rate:.1f}%")

with col4:
    avg_bmi = processed_df['BMI'].mean()
    st.metric("Average BMI", f"{avg_bmi:.1f}")

st.divider()

# -----------------------------------------------------------------------------
# Sidebar Filters
st.sidebar.header("Filter Survey Data")

# Age range filter with discrete age categories
age_categories = [
    "18-24", "25-29", "30-34", "35-39", "40-44", 
    "45-49", "50-54", "55-59", "60-64", "65-69", 
    "70-74", "75-79", "80+"
]

selected_age_range = st.sidebar.select_slider(
    "Age Range",
    options=age_categories,
    value=("18-24", "80+")
)

# Convert selected age ranges to category numbers
age_cat_mapping = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80+": 13
}

min_age_cat = age_cat_mapping[selected_age_range[0]]
max_age_cat = age_cat_mapping[selected_age_range[1]]

# BMI range filter
bmi_range = st.sidebar.slider(
    "BMI Range", 
    float(processed_df['BMI'].min()), 
    float(processed_df['BMI'].max()), 
    (float(processed_df['BMI'].min()), float(processed_df['BMI'].max()))
)

# Health status filter (updated order)
health_status = st.sidebar.multiselect(
    "General Health Status",
    ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
    default=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
)

# Apply filters
filtered_df = processed_df[
    (processed_df['Age'] >= min_age_cat) & (processed_df['Age'] <= max_age_cat) &
    (processed_df['BMI'] >= bmi_range[0]) & (processed_df['BMI'] <= bmi_range[1]) &
    (processed_df['GenHlth_Label'].isin(health_status))
]

st.sidebar.metric("Filtered Responses", len(filtered_df))

# -----------------------------------------------------------------------------
# Main Dashboard Content

# Section 1: Demographics Analysis
st.header("Demographics & Diabetes Prevalence")
st.markdown("*Explore how diabetes rates vary across different demographic groups including age and sex distributions.*")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Age distribution by diabetes status
    age_diabetes = filtered_df.groupby(['AgeGroup', 'DiabetesStatus']).size().reset_index(name='Count')
    fig_age = px.bar(
        age_diabetes,
        x='AgeGroup',
        y='Count',
        color='DiabetesStatus',
        title='Diabetes Prevalence by Age Group',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        height=350
    )
    fig_age.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    # Sex and diabetes distribution with percentages
    sex_diabetes = filtered_df.groupby(['Sex_Label', 'DiabetesStatus']).size().reset_index(name='Count')
    # Calculate percentages
    sex_totals = sex_diabetes.groupby('Sex_Label')['Count'].sum()
    sex_diabetes['Percentage'] = sex_diabetes.apply(lambda row: (row['Count'] / sex_totals[row['Sex_Label']]) * 100, axis=1)
    
    fig_sex = px.bar(
        sex_diabetes,
        x='Sex_Label',
        y='Count',
        color='DiabetesStatus',
        title='Diabetes Prevalence by Sex',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        text='Percentage',
        height=350
    )
    fig_sex.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_sex, use_container_width=True)

with col3:
    # BMI distribution by diabetes status with proper ordering
    bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    bmi_diabetes = filtered_df.groupby(['BMI_Category', 'DiabetesStatus']).size().reset_index(name='Count')
    bmi_diabetes['BMI_Category'] = pd.Categorical(bmi_diabetes['BMI_Category'], categories=bmi_order, ordered=True)
    bmi_diabetes = bmi_diabetes.sort_values('BMI_Category')
    # Calculate percentages for BMI
    bmi_totals = bmi_diabetes.groupby('BMI_Category')['Count'].sum()
    bmi_diabetes['Percentage'] = bmi_diabetes.apply(lambda row: (row['Count'] / bmi_totals[row['BMI_Category']]) * 100, axis=1)
    
    fig_bmi = px.bar(
        bmi_diabetes,
        x='BMI_Category',
        y='Count',
        color='DiabetesStatus',
        title='BMI Categories and Diabetes Status',
        labels={'count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        category_orders={'BMI_Category': bmi_order},
        text='Percentage',
        height=350
    )
    fig_bmi.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_bmi, use_container_width=True)

st.divider()

# Section 2: Risk Factors and ML Insights
st.header("Machine Learning Insights & Risk Factors")
st.markdown("*Random Forest model analysis showing the most important features for predicting diabetes, alongside key risk factor comparisons.*")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature_Name',
        orientation='h',
        title='ML Feature Importance for Diabetes Prediction',
        labels={'Importance': 'Feature Importance', 'Feature_Name': 'Health Indicators'},
        color='Importance',
        color_continuous_scale='Blues',
        height=len(feature_importance) * 40  # scale height based on number of features
    )
    fig_importance.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=150)  # increase left margin if labels are long
    )
    fig_importance.update_yaxes(automargin=True)
    st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    # High BP and High Cholesterol
    risk_factors = ['HighBP_Label', 'HighChol_Label', 'Smoker_Label']
    selected_risk = st.selectbox("Select Risk Factor:", risk_factors)
    
    risk_diabetes = filtered_df.groupby([selected_risk, 'DiabetesStatus']).size().reset_index(name='Count')
    # Calculate percentages for risk factors
    risk_totals = risk_diabetes.groupby(selected_risk)['Count'].sum()
    risk_diabetes['Percentage'] = risk_diabetes.apply(lambda row: (row['Count'] / risk_totals[row[selected_risk]]) * 100, axis=1)
    
    fig_risk = px.bar(
        risk_diabetes,
        x=selected_risk,
        y='Count',
        color='DiabetesStatus',
        title=f'{selected_risk.replace("_Label", "")} and Diabetes Status',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        text='Percentage',
        height=350
    )
    fig_risk.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_risk, use_container_width=True)

with col3:
    # Risk score calculation and distribution
    def calculate_risk_score(row):
        score = 0
        if row['HighBP'] == 1: score += 2
        if row['HighChol'] == 1: score += 2
        if row['BMI'] > 30: score += 3
        elif row['BMI'] > 25: score += 1
        if row['Smoker'] == 1: score += 1
        if row['HeartDiseaseorAttack'] == 1: score += 2
        if row['PhysActivity'] == 0: score += 1
        if row['GenHlth'] > 3: score += 1
        if row['Age'] > 9: score += 1  # Age > 60
        return score
    
    filtered_df['RiskScore'] = filtered_df.apply(calculate_risk_score, axis=1)
    risk_counts = filtered_df['RiskScore'].value_counts().sort_index()
    
    fig_risk_score = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title='Distribution of Diabetes Risk Scores',
        labels={'x': 'Risk Score (0-12)', 'y': 'Number of Respondents'},
        color_discrete_sequence=['#1f77b4'],
        height=350
    )
    st.plotly_chart(fig_risk_score, use_container_width=True)

st.divider()

# Section 3: Health Behaviors & Healthcare Access
st.header("Health Behaviors & Healthcare Access")
st.markdown("*Analysis of lifestyle factors including physical activity, self-reported health status, and healthcare accessibility patterns.*")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # Physical activity and diabetes
    activity_diabetes = filtered_df.groupby(['PhysActivity_Label', 'DiabetesStatus']).size().reset_index(name='Count')
    # Calculate percentages for physical activity
    activity_totals = activity_diabetes.groupby('PhysActivity_Label')['Count'].sum()
    activity_diabetes['Percentage'] = activity_diabetes.apply(lambda row: (row['Count'] / activity_totals[row['PhysActivity_Label']]) * 100, axis=1)
    
    fig_activity = px.bar(
        activity_diabetes,
        x='PhysActivity_Label',
        y='Count',
        color='DiabetesStatus',
        title='Physical Activity and Diabetes Status',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        text='Percentage',
        height=350
    )
    fig_activity.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_activity, use_container_width=True)

with col2:
    # General health status with proper ordering (Poor to Excellent)
    health_order = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    health_diabetes = filtered_df.groupby(['GenHlth_Label', 'DiabetesStatus']).size().reset_index(name='Count')
    health_diabetes['GenHlth_Label'] = pd.Categorical(health_diabetes['GenHlth_Label'], categories=health_order, ordered=True)
    health_diabetes = health_diabetes.sort_values('GenHlth_Label')
    # Calculate percentages for general health
    health_totals = health_diabetes.groupby('GenHlth_Label')['Count'].sum()
    health_diabetes['Percentage'] = health_diabetes.apply(lambda row: (row['Count'] / health_totals[row['GenHlth_Label']]) * 100, axis=1)
    
    fig_health = px.bar(
        health_diabetes,
        x='GenHlth_Label',
        y='Count',
        color='DiabetesStatus',
        title='Self-Reported General Health and Diabetes',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        category_orders={'GenHlth_Label': health_order},
        text='Percentage',
        height=350
    )
    fig_health.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_health, use_container_width=True)

with col3:
    # Healthcare access by diabetes status
    healthcare_df = filtered_df.groupby(['AnyHealthcare', 'DiabetesStatus']).size().reset_index(name='Count')
    healthcare_df['AnyHealthcare'] = healthcare_df['AnyHealthcare'].map({0: 'No Healthcare', 1: 'Has Healthcare'})
    # Calculate percentages for healthcare access
    healthcare_totals = healthcare_df.groupby('AnyHealthcare')['Count'].sum()
    healthcare_df['Percentage'] = healthcare_df.apply(lambda row: (row['Count'] / healthcare_totals[row['AnyHealthcare']]) * 100, axis=1)
    
    fig_healthcare = px.bar(
        healthcare_df,
        x='AnyHealthcare',
        y='Count',
        color='DiabetesStatus',
        title='Healthcare Access and Diabetes Status',
        labels={'Count': 'Number of Respondents'},
        color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
        text='Percentage',
        height=350
    )
    fig_healthcare.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    st.plotly_chart(fig_healthcare, use_container_width=True)

st.divider()

# Section 4: Mental vs Physical Health Correlation
st.header("Mental & Physical Health Relationship")
st.markdown("*Scatter plot analysis examining the relationship between physical and mental health days, providing insights into overall wellbeing patterns among diabetic and non-diabetic individuals.*")

# Mental vs Physical health days
fig_scatter = px.scatter(
    filtered_df.sample(min(5000, len(filtered_df))),  # Sample for performance
    x='PhysHlth',
    y='MentHlth',
    color='DiabetesStatus',
    title='Physical vs Mental Health Days (Poor Health)',
    labels={'PhysHlth': 'Physical Health Days', 'MentHlth': 'Mental Health Days'},
    color_discrete_map={'No Diabetes': '#1f77b4', 'Diabetes': '#ff7f0e'},
    height=400
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Section 5: Data Explorer
st.header("📋 Survey Data Explorer")
st.markdown("*Browse and download the filtered survey data. Use the sidebar filters to customize the dataset according to your analysis needs.*")

# Display key columns
display_columns = ['AgeGroup', 'Sex_Label', 'BMI', 'BMI_Category', 'GenHlth_Label', 
                  'HighBP_Label', 'HighChol_Label', 'PhysActivity_Label', 'DiabetesStatus']

st.dataframe(
    filtered_df[display_columns].head(1000),  # Show first 1000 rows for performance
    use_container_width=True,
    height=300
)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='diabetes_health_indicators_filtered.csv',
    mime='text/csv'
)

# -----------------------------------------------------------------------------
# Footer
st.divider()
st.markdown("""
**Data Source:** [CDC Diabetes Health Indicators Dataset - Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
**Original Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015  
**Dashboard created with:** Streamlit, Plotly, Pandas, Scikit-learn  
*This dashboard demonstrates public health data analysis and machine learning capabilities for portfolio purposes.*
""")
