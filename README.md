# student-mental-health-dashboard

🧠 Student Mental Health Dashboard
This repository contains an interactive data dashboard project focused on analyzing the mental health of students using data analytics and visualization techniques. It was developed as part of an assessment from Structured Labs.

The application leverages:

* Pandas & NumPy for data manipulation,
* Scikit-learn for ML modeling (K-Means, Random Forest),
* Plotly for rich visualizations,
* Preswald – a Python framework for building interactive dashboards.

📊 Project Overview
The Student Mental Health Dashboard explores the impact of various factors such as sleep, exercise, diet, social media usage, financial and relationship stress on student mental health. The project presents findings through an interactive web interface powered by Preswald.

🔍 Key Features
Data Cleaning & Preprocessing: Handles missing values, duplicates, and transforms features like age groups and stress categories.

* Feature Engineering:
  Wellness Score combining sleep, exercise, and diet.
  Study Efficiency = GPA / Study Hours.
  Stress Factor Score aggregating multiple stress sources.

* Visual Analytics:
  Correlation matrix between lifestyle factors and mental stress.
  Histograms, box plots, and scatter plots grouped by gender and age group.
  Predictive Modeling (using Random Forest & K-Means for insight generation).

* Insights & Recommendations:
  Highlights how sleep, exercise, and social media usage correlate with stress levels.
  Suggests targeted interventions like sleep hygiene workshops and academic support.

🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Plotly
* Preswald

📁 Dataset
The project uses a student mental health dataset (Studentsmentalhealth_csv) accessed from Kaggle. This dataset includes features like:
* Mental Stress Level
* Sleep Duration
* GPA
* Physical Exercise
* Social Media Usage
* Family and Relationship Stress

🧠 Example Insights
* Students with less than 6 hours of sleep report significantly higher stress levels.
* Regular physical activity is associated with lower stress.
* High social media usage tends to correlate with higher mental stress.
* Study efficiency (GPA per study hour) is a more reliable indicator than raw study time.

📎 SQL Integration
A basic SQL query feature is available to extract filtered records from a student mental health database, such as identifying high-risk students based on depression and anxiety scores.
