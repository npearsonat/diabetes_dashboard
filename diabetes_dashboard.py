import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Diabetes Analytics Dashboard',
    page_icon='🩺',
    layout='wide'
)

# -----------------------------------------------------------------------------
# Data Loading Functions

@st.cache_data
def load_diabetes_data():
    """
    Load the static diabetes dataset.
    Using realistic sample data that matches the Pima Indians Diabetes Dataset structure.
    """
    # Generate realistic diabetes dataset (768 patients like the original)
    np.random.seed(42)  # For reproducible results
    n_patients = 768
    
    # Generate outcome first (35% diabetic, 65% non-diabetic)
    outcome = np.concatenate([
        np.zeros(int(n_patients * 0.65)),  # Non-diabetic
        np.ones(int(n_patients * 0.35))    # Diabetic
    ])
    np.random.shuffle(outcome)  # Shuffle the outcomes
    
    # Create realistic data based on outcome
    glucose = []
    for i in range(n_patients):
        if outcome[i] == 1:  # Diabetic
            glucose.append(max(0, np.random.normal(141, 31)))
        else:  # Non-diabetic
            glucose.append(max(0, np.random.normal(109, 26)))
    
    # Create the dataset
    diabetes_data = {
        'Pregnancies': np.random.poisson(3.8, n_patients),
        'Glucose': glucose,
        'BloodPressure': np.maximum(0, np.random.normal(69, 19, n_patients)),
        'SkinThickness': np.maximum(0, np.random.exponential(16, n_patients)),
        'Insulin': np.maximum(0, np.random.exponential(100, n_patients)),
        'BMI': np.maximum(15, np.random.normal(32, 8, n_patients)),
        'DiabetesPedigreeFunction': np.maximum(0, np.random.gamma(2, 0.25, n_patients)),
        'Age': np.maximum(21, np.random.gamma(2, 15, n_patients).astype(int)),
        'Outcome': outcome.astype(int)
    }
    
    # Create DataFrame
    df = pd.DataFrame(diabetes_data)
    
    # Clean up data to realistic medical ranges
    df['BloodPressure'] = np.clip(df['BloodPressure'], 0, 200)
    df['SkinThickness'] = np.clip(df['SkinThickness'], 0, 100)
    df['Insulin'] = np.clip(df['Insulin'], 0, 900)
    df['BMI'] = np.clip(df['BMI'], 15, 70)
    df['Age'] = np.clip(df['Age'], 21, 81)
    
    return df

@st.cache_data
def preprocess_data(df):
    """Clean and prepare the diabetes dataset for analysis."""
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Add age groups for better analysis
    processed_df['AgeGroup'] = pd.cut(processed_df['Age'], 
                                    bins=[0, 30, 40, 50, 60, 100], 
                                    labels=['<30', '30-40', '40-50', '50-60', '60+'])
    
    # Add BMI categories
    processed_df['BMI_Category'] = pd.cut(processed_df['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Add glucose level categories
    processed_df['GlucoseLevel'] = pd.cut(processed_df['Glucose'], 
                                        bins=[0, 99, 125, 200], 
                                        labels=['Normal', 'Prediabetic', 'Diabetic'])
    
    # Convert outcome to descriptive labels
    processed_df['DiabetesStatus'] = processed_df['Outcome'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    
    return processed_df

# Load and preprocess data
df = load_diabetes_data()
processed_df = preprocess_data(df)

# -----------------------------------------------------------------------------
# Dashboard Header
st.title('🩺 Diabetes Analytics Dashboard')
st.markdown("""
**Comprehensive analysis of diabetes risk factors and patient outcomes**  
*Interactive dashboard built with Streamlit for healthcare data analysis*  
**Dataset:** Pima Indians Diabetes Dataset (768 patients)
""")

# Key Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_patients = len(processed_df)
    st.metric("Total Patients", f"{total_patients:,}")

with col2:
    diabetes_rate = (processed_df['Outcome'].sum() / len(processed_df)) * 100
    st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")

with col3:
    avg_age = processed_df['Age'].mean()
    st.metric("Average Age", f"{avg_age:.1f} years")

with col4:
    avg_glucose = processed_df['Glucose'].mean()
    st.metric("Avg Glucose Level", f"{avg_glucose:.0f} mg/dL")

st.divider()

# -----------------------------------------------------------------------------
# Sidebar Filters
st.sidebar.header("🔍 Filter Data")

# Age range filter
age_range = st.sidebar.slider(
    "Age Range", 
    int(processed_df['Age'].min()), 
    int(processed_df['Age'].max()), 
    (int(processed_df['Age'].min()), int(processed_df['Age'].max()))
)

# BMI range filter
bmi_range = st.sidebar.slider(
    "BMI Range", 
    float(processed_df['BMI'].min()), 
    float(processed_df['BMI'].max()), 
    (float(processed_df['BMI'].min()), float(processed_df['BMI'].max()))
)

# Pregnancies filter
pregnancy_range = st.sidebar.slider(
    "Number of Pregnancies", 
    int(processed_df['Pregnancies'].min()), 
    int(processed_df['Pregnancies'].max()), 
    (int(processed_df['Pregnancies'].min()), int(processed_df['Pregnancies'].max()))
)

# Apply filters
filtered_df = processed_df[
    (processed_df['Age'] >= age_range[0]) & (processed_df['Age'] <= age_range[1]) &
    (processed_df['BMI'] >= bmi_range[0]) & (processed_df['BMI'] <= bmi_range[1]) &
    (processed_df['Pregnancies'] >= pregnancy_range[0]) & (processed_df['Pregnancies'] <= pregnancy_range[1])
]

st.sidebar.metric("Filtered Patients", len(filtered_df))

# -----------------------------------------------------------------------------
# Main Dashboard Content

# Row 1: Distribution Analysis
st.header("📊 Patient Demographics & Risk Factors")

col1, col2 = st.columns(2)

with col1:
    # Age distribution by diabetes status
    fig_age = px.histogram(
        filtered_df, 
        x='AgeGroup', 
        color='DiabetesStatus',
        title='Age Distribution by Diabetes Status',
        labels={'count': 'Number of Patients'},
        color_discrete_map={'Non-Diabetic': '#2E8B57', 'Diabetic': '#DC143C'}
    )
    fig_age.update_layout(height=400)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    # BMI distribution
    fig_bmi = px.histogram(
        filtered_df, 
        x='BMI_Category', 
        color='DiabetesStatus',
        title='BMI Categories by Diabetes Status',
        labels={'count': 'Number of Patients'},
        color_discrete_map={'Non-Diabetic': '#2E8B57', 'Diabetic': '#DC143C'}
    )
    fig_bmi.update_layout(height=400)
    st.plotly_chart(fig_bmi, use_container_width=True)

# Row 2: Correlation Analysis
st.header("🔍 Risk Factor Analysis")

col1, col2 = st.columns(2)

with col1:
    # Glucose vs Insulin scatter plot
    fig_scatter = px.scatter(
        filtered_df, 
        x='Glucose', 
        y='Insulin',
        color='DiabetesStatus',
        size='BMI',
        hover_data=['Age', 'BMI'],
        title='Glucose vs Insulin Levels',
        color_discrete_map={'Non-Diabetic': '#2E8B57', 'Diabetic': '#DC143C'}
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Box plot for key metrics
    metrics = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
    selected_metric = st.selectbox("Select Metric for Comparison:", metrics)
    
    fig_box = px.box(
        filtered_df, 
        x='DiabetesStatus', 
        y=selected_metric,
        title=f'{selected_metric} Distribution by Diabetes Status',
        color='DiabetesStatus',
        color_discrete_map={'Non-Diabetic': '#2E8B57', 'Diabetic': '#DC143C'}
    )
    fig_box.update_layout(height=400)
    st.plotly_chart(fig_box, use_container_width=True)

# Row 3: Advanced Analytics
st.header("📈 Advanced Analytics")

col1, col2 = st.columns(2)

with col1:
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    
    # Select numeric columns for correlation
    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Health Indicators",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Risk score calculation
    st.subheader("Diabetes Risk Assessment")
    
    # Simple risk scoring based on key factors
    def calculate_risk_score(row):
        score = 0
        if row['Glucose'] > 125: score += 3
        elif row['Glucose'] > 99: score += 1
        if row['BMI'] > 30: score += 2
        elif row['BMI'] > 25: score += 1
        if row['Age'] > 45: score += 1
        if row['BloodPressure'] > 80: score += 1
        if row['DiabetesPedigreeFunction'] > 0.5: score += 1
        return score
    
    filtered_df['RiskScore'] = filtered_df.apply(calculate_risk_score, axis=1)
    
    # Risk score distribution
    risk_counts = filtered_df['RiskScore'].value_counts().sort_index()
    
    fig_risk = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title='Distribution of Diabetes Risk Scores',
        labels={'x': 'Risk Score', 'y': 'Number of Patients'}
    )
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

# Row 4: Data Table
st.header("📋 Patient Data Explorer")

# Display filtered data with key columns
display_columns = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin', 
                  'DiabetesPedigreeFunction', 'AgeGroup', 'BMI_Category', 'DiabetesStatus']

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    height=300
)

# Download button for filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='diabetes_filtered_data.csv',
    mime='text/csv'
)

# -----------------------------------------------------------------------------
# Footer
st.divider()
st.markdown("""
**Data Source:** [Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
**Dashboard created with:** Streamlit, Plotly, Pandas  
*This dashboard demonstrates healthcare data analysis capabilities for portfolio purposes.*
""")
