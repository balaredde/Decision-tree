# Salary Prediction using Decision Trees  

This repository contains a Jupyter Notebook that demonstrates how to predict whether an employee earns more than **$100K** based on company, job role, and degree. The classification is performed using **Decision Trees** from the `scikit-learn` library.  

## Dataset  
The dataset `salaries.csv` contains the following columns:  
- `company` - Name of the company  
- `job` - Job role  
- `degree` - Type of degree held  
- `salary_more_than_100k` - Target variable (1 = Yes, 0 = No)  

---

## Installation  

To run this project, install the required dependencies using:  

```bash
pip install pandas scikit-learn
```

---

## Usage  

Clone the repository and open the Jupyter Notebook to explore the code and experiment with different inputs.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
jupyter notebook
```

---

## Code Overview  

### Importing Necessary Libraries  

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
```

### Loading the Dataset  

```python
df = pd.read_csv('salaries.csv')
df.head()
```

### Preprocessing the Data  

- Dropping the target column (`salary_more_than_100k`) from input features  
- Encoding categorical variables (`company`, `job`, `degree`) into numerical format  

```python
inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
```

### Training the Decision Tree Model  

```python
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
```

### Making Predictions  

Predicting whether an employee with specific attributes earns more than **$100K**:  

```python
# Example: Employee from company 2, job role 2, and degree 1
prediction = model.predict([[2, 2, 1]])
print("Prediction:", prediction)
```

```python
# Example: Employee from company 2, job role 0, and degree 1
model.predict([[2, 0, 1]])

# Example: Employee from company 2, job role 0, and degree 0
model.predict([[2, 0, 0]])
```

---

## Results  

- The model predicts whether a person earns **more than $100K** based on company, job, and degree.  
- Categorical data is encoded into numerical format for processing.  
- The Decision Tree model is used for classification.  

---

## Contributing  

Feel free to fork this repository, create a feature branch, and submit a pull request! 🚀  

---

## License  

This project is open-source and available under the **MIT License**.  
