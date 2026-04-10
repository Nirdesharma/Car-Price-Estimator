# Car-Price-Estimator
📌 Overview

This project aims to predict the price of Ford cars using Machine Learning techniques. It covers the complete pipeline from data collection to model building, focusing on improving prediction accuracy through proper preprocessing techniques.

📂 Dataset
Dataset: Ford Car Dataset
Features include:
Model
Year
Price
Transmission
Mileage
Fuel Type
Engine Size
⚙️ Technologies Used
Python
Pandas
NumPy
Matplotlib / Seaborn
Scikit-learn
🔍 Project Workflow
1. Data Collection
Collected Ford car dataset containing various car attributes.
2. Exploratory Data Analysis (EDA)
Analyzed feature distributions
Identified relationships between variables
Detected outliers and patterns
3. Data Cleaning & Preprocessing
Handled missing values
Removed inconsistencies
Applied encoding techniques:
Label Encoding
One-Hot Encoding
🤖 Model Building
Algorithm used:
Linear Regression
Built two versions of the model:
Using Label Encoding
Using One-Hot Encoding
📈 Results & Insights
Encoding Technique	Accuracy
Label Encoding	73%
One-Hot Encoding	84%

✅ Final Model Selected:
The model trained using One-Hot Encoding was selected because it provided significantly better performance.
