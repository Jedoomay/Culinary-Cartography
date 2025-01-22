Culinary Cartography
Culinary Cartography is a data analysis and machine learning project designed to explore global culinary trends, their ingredients, and correlations with socioeconomic factors such as health scores, life expectancy, GDP, and IQ. The project leverages interactive visualizations and a classification model to analyze recipes and predict their regional origins.

Table of Contents
Project Overview
Key Features
Technologies Used
Setup and Installation
How to Use
Dataset Overview
Project Structure
Visualizations and Results
Future Work
Acknowledgements
Project Overview
Culinary Cartography examines how culinary ingredients vary across regions and explores correlations between recipes and factors such as:

Health scores by region.
Socioeconomic indicators like GDP per capita, IQ, and life expectancy.
Ingredient frequency and network relationships.
The project includes an interactive dashboard and a machine learning model to predict the region of origin for recipes based on their ingredients.

Key Features
Data Analysis:

Explores ingredient frequency and its correlation with regional statistics.
Visualizes socioeconomic and health trends using graphs and maps.
Machine Learning:

Logistic Regression model trained to classify recipes based on their ingredients.
Achieved a precision score of 0.8 across 20+ regions.
Interactive Dashboard:

Visualizes data using:
Ingredient frequency heatmaps.
Correlation plots (e.g., GDP vs. Calorie Supply).
Health scores by region.
Confusion matrix and classification reports.
Technologies Used
Programming Languages:

Python
Libraries and Frameworks:

Data Manipulation: pandas, numpy
Visualization: dash, plotly, matplotlib
Machine Learning: scikit-learn, statsmodels
Graph Analysis: networkx
Web Framework: Dash, dash-bootstrap-components
Design:

Custom CSS styling for responsive and user-friendly layouts.
Setup and Installation
Prerequisites
Python 3.7 or higher
Pip package manager
Installation Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/YourUsername/culinary-cartography.git
cd culinary-cartography
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
python main.py
Open your browser and navigate to:

arduino
Copy
Edit
http://127.0.0.1:8050/
How to Use
Home Page: Learn about the project and its purpose.
Ingredient Network Graph: Explore relationships between ingredients for specific recipes.
Ingredient Frequency Heatmap: Compare the frequency of ingredients across regions.
Country Statistics Map: Visualize correlations between socioeconomic metrics and calories.
Health Scores by Region: View health score trends with bar charts.
Model Performance: Analyze the machine learning model's confusion matrix and classification report.
Dataset Overview
Datasets Used:

recipes_with_region.csv: Recipes with associated regions and ingredients.
health_scores_of_regions.csv: Health scores for various regions.
combined_aspects.csv: Socioeconomic statistics such as GDP, IQ, and life expectancy.
averages_data.csv: Data for correlation visualizations.
recipe.csv: Ingredient details for specific recipes.
Data Preprocessing:

Ingredients were binarized into a sparse matrix for machine learning.
Regions with low representation were excluded.
Correlation plots were generated using cleaned and aggregated data.
Project Structure
graphql
Copy
Edit
culinary-cartography/
│
├── main.py                    # Main application code
├── data/
│   ├── recipes_with_region.csv # Recipes with regional tags
│   ├── health_scores_of_regions.csv # Health score data
│   ├── combined_aspects.csv   # Socioeconomic statistics
│   ├── averages_data.csv      # Processed correlation data
│   └── recipe.csv             # Ingredient details
│
├── assets/
│   └── style.css              # Custom CSS for dashboard styling
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
Visualizations and Results
Ingredient Frequency Heatmap:

Displays the relative frequency of ingredients across regions.
Correlation Graphs:

Shows relationships such as:
GDP vs. Calorie Supply
IQ vs. Calorie Supply
Life Expectancy vs. Calorie Supply
Health Scores by Region:

Bar chart of average health scores for each region.
Model Performance:

Confusion matrix and classification report for the recipe classification model.
Future Work
Integrate more datasets:

Climate data and ingredient availability by region.
Nutritional values for each ingredient.
Enhance the machine learning pipeline:

Experiment with ensemble models for improved classification.
Add explainability methods like SHAP or LIME.
Expand the dashboard:

Real-time data updates.
Interactive filtering for deeper analysis.
Acknowledgements
This project was developed as part of the PROJ 201 course by:

Emirhan Oğuz
Ahmet Ergin
We thank our professors and peers for their guidance and support.
