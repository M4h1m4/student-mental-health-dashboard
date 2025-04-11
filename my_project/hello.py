from preswald import text, plotly, connect, get_df, table, slider, query
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

text("# Student Mental Health Dashboard")
text("This is an interactive dashboard that explores the relationship between student mental health and various factors such as lifestyle factors, stress levels and coping mechanisms.")

# Load the CSV
connect() # load in all sources, which by default is the sample_csv
df = get_df('Studentsmentalhealth_csv')
df.columns = df.columns.str.strip()

#Data Overview
text("### Data Overview")
text(f"The dataset contains {df.shape[0]} student records with {df.shape[1]} features.")
#text(f"datasetcolums{df.columns}")
table(df.head(10))

#Data Cleaning
text("### Data Cleaning")

#check for missing values
missing_values = df.isnull().sum()
missing_df = pd.DataFrame({'Column':missing_values.index,'Missing Count':missing_values.values})
missing_df = missing_df[missing_df['Missing Count'] > 0]

#missing_df = missing_df.sort_values(by='Missing Count', ascending=False)
text(f"The dataset contains {len(missing_df)} columns with missing values.")

#Just in case if there are any missing values
if not missing_df.empty:
    #text("### Missing Values in Dataset")
    table(missing_df)

initial_rows = df.shape[0]
df = df.drop_duplicates('Student ID')
#text(f"- Removed {initial_rows - df.shape[0]} duplicate student records")


# Handle missing values for numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
        text(f"- Filled missing values in '{col}' with median value")

# Handle missing values for categorical columns
categorical_cols = ['Gender', 'Counseling Attendance', 'Family Mental Health History', 'Medical Condition']
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
        text(f"- Filled missing values in '{col}' with mode value")

df['Age Group'] = pd.cut(df['Age'], bins=[17, 20, 23, 26, 31], labels=['18-20','21-23','24-26','27-30'])

text("### Feature Engineering")
text("Creating derived features to enhance analysis:")


df['Stress Category']=pd.cut(df['Mental Stress Level'],bins=[0,3,7,10],labels=['Low(1-3)','Moderate(4-7)','High(8-10)'])

#Calculating Efficiency ratio
df['Study Efficiency'] = df['Academic Performance (GPA)']/df['Study Hours Per Week']
df['Study Efficiency'] = df['Study Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(df['Study Efficiency'].median())
text("- Created 'Study Efficiency' metric (GPA per study hour)")

#Creating Wellness Score
df['Wellness Score'] = ((df['Sleep Duration (Hours per night)']/8) * 0.4 + (df['Physical Exercise (Hours per week)']/10)*0.3 + (df['Diet Quality']/5)*0.3) * 10 # scale 1-10

text("- Created Wellness Score based on Sleep , Exercise, and Diet Quality")

#Creating Stress Factor Score
stres_columns =[
    'Financial Stress',
    'Peer Pressure',
    'Relationship Stress',
    'Cognitive Distortions',
]
df['Stress Factor Score'] = df[stres_columns].sum(axis=1)
text("- Created Cummulative Stress Factor Score")

text("### Key Mental Health Metrics")
#creating metrics display

avg_stress = df['Mental Stress Level'].mean()
avg_wellness = df['Wellness Score'].mean()
high_stress_pct = (df['Mental Stress Level'] > 8).mean() * 100
avg_sleep = df['Sleep Duration (Hours per night)'].mean()

text(f"- **Average Mental Stress Level** : {avg_stress:.2f}/10 ")
text(f"- **Average Wellness Score** : {avg_wellness:.2f}/10 ")
text(f"- **Students with High Stress (>8)** : {high_stress_pct:.2f}% ")
text(f"- **Average Sleep Duration** : {avg_sleep:.2f} hours per night")


#Mental Stress Distribution

#Stress by Gender
stress_fig = px.histogram(
    df, x='Mental Stress Level', 
    color='Gender', marginal='box', 
    title='Distribution of Mental Stress Level', 
    labels={'Mental Stress Level':'Stress Level(1-10)'})

stress_fig.update_layout(bargap=0.1)
plotly(stress_fig)


age_stress = df.groupby('Age Group')['Mental Stress Level'].mean().reset_index()
age_fig = px.bar(
    age_stress, x='Age Group', y='Mental Stress Level', title ='Average Stress by Age Group', color='Age Group'
)
plotly(age_fig)

#Correlation Analysis

text("### Factor Correlation Analysis")
text("Examining relationship between lifestyle factors and mental stress")

corr_cols =[
    'Mental Stress Level', 'Age', 'Academic Performance (GPA)', 
    'Study Hours Per Week', 'Social Media Usage (Hours per day)', 
    'Sleep Duration (Hours per night)', 'Physical Exercise (Hours per week)',
    'Family Support', 'Financial Stress', 
    'Peer Pressure', 'Relationship Stress',
    'Diet Quality', 'Cognitive Distortions', 
    'Substance Use'
]

existing_cols = [col for col in corr_cols if col in df.columns]
corr_matrix = df[existing_cols].corr()

heatmap = px.imshow(
    corr_matrix, 
    text_auto = True, 
    color_continuous_scale = 'RdBu_r', 
    aspect = 'auto', 
    title = 'Correlation Matrix of Mental Health Factors'
)
plotly(heatmap)

text("Key Factor Relationships")

#Sleep vs Stress
sleep_fig = px.scatter(
    df,
    x='Sleep Duration (Hours per night)',
    y='Mental Stress Level',
    color = 'Gender',
    title = 'Sleep Duration vs Mental Stress Level',
    labels = {
        'Sleep Duration (Hours Per Night)':'Sleep Hours',
        'Mental Stress Level':'Stress Level(1-10)'
    }
)
plotly(sleep_fig)

excercise_fig = px.scatter(
    df,
    x='Physical Exercise (Hours per week)',
    y='Mental Stress Level',
    color = 'Gender',
    #trendline = 'ols',
    title = 'Excercise vs Mental Stress',
    labels = {
        'Physical Exercise (Hours per week)':'Exercise Hours Per Week',
        'Mental Stress Level':'Stress Level(1-10)'
        }
)
plotly(excercise_fig)

# Social Media Usage vs Mental Stress (Box Plot)
social_box = px.box(
    df,
    x='Social Media Usage (Hours per day)',
    y='Mental Stress Level',
    color='Gender',
    points='all',
    title='Mental Stress Level by Social Media Usage',
    labels={
        'Social Media Usage (Hours per day)': 'Social Media Hours per Day',
        'Mental Stress Level': 'Stress Level (1-10)'
    }
)
plotly(social_box)

# GPA vs Mental Stress (Box Plot)
gpa_box = px.box(
    df,
    x='Academic Performance (GPA)',
    y='Mental Stress Level',
    color='Gender',
    points='all',
    title='Mental Stress Level by Academic Performance (GPA)',
    labels={
        'Academic Performance (GPA)': 'GPA (0-4.0)',
        'Mental Stress Level': 'Stress Level (1-10)'
    }
)
plotly(gpa_box)


text("## Key Insights & Recommendations")

# Generate insights based on analysis
text("### Primary Findings")

sleep_corr = corr_matrix.loc['Sleep Duration (Hours per night)', 'Mental Stress Level'] if 'Sleep Duration (Hours per night)' in corr_matrix.index else 0
text(f"1. **Sleep Impact**: Analysis shows a correlation of {sleep_corr:.2f} between sleep duration and stress levels. " +
     f"Students with less than 6 hours of sleep have {(df[df['Sleep Duration (Hours per night)'] < 6]['Mental Stress Level'].mean() - df[df['Sleep Duration (Hours per night)'] >= 7]['Mental Stress Level'].mean()):.1f} points higher stress on average.")

exercise_corr = corr_matrix.loc['Physical Exercise (Hours per week)', 'Mental Stress Level'] if 'Physical Exercise (Hours per week)' in corr_matrix.index else 0
text(f"2. **Exercise Benefit**: Regular physical exercise shows a correlation of {exercise_corr:.2f} with stress levels. " +
     f"Students with 5+ hours weekly exercise show {(df[df['Physical Exercise (Hours per week)'] < 3]['Mental Stress Level'].mean() - df[df['Physical Exercise (Hours per week)'] >= 5]['Mental Stress Level'].mean()):.1f} points lower stress.")

social_corr = corr_matrix.loc['Social Media Usage (Hours per day)', 'Mental Stress Level'] if 'Social Media Usage (Hours per day)' in corr_matrix.index else 0
text(f"3. **Social Media Usage**: Higher social media usage has a correlation of {social_corr:.2f} with stress levels. " +
     f"Students using social media {df['Social Media Usage (Hours per day)'].mean():.1f}+ hours daily show significantly higher stress.")

study_corr = corr_matrix.loc['Study Efficiency', 'Mental Stress Level'] if 'Study Efficiency' in corr_matrix.index else 0
text(f"4. **Study Efficiency**: Study efficiency has a correlation of {study_corr:.2f} with stress levels, " +
     "suggesting quality of study time may be more important than quantity.")

text("### Recommended Interventions")
text("1. **Sleep Education Program**: Implement workshops on sleep hygiene and its impact on mental health.")
text("2. **Physical Activity Initiative**: Develop accessible fitness programs targeting high-stress students.")
text("3. **Digital Wellness Campaign**: Create awareness about healthy social media usage patterns.")
text("4. **Academic Support**: Design interventions focusing on study efficiency rather than just study hours.")
text("5. **Mental Health Resources**: Expand counseling services, especially for students with family history of mental health issues.")


sql = " SELECT * FROM student_mental_health WHERE depression_score > 50 AND anxiety_score > 40 "
filtered_df = query(sql, "student_mental_health")