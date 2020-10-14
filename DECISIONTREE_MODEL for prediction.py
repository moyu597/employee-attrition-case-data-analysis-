# import modules
import pandas as pd  # for dataframes
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for plotting graphs
from sklearn import preprocessing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

excelfile = pd.ExcelFile('real.xlsx')
data = excelfile.parse('Sheet1')

le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary'] = le.fit_transform(data['salary'])
data['dept'] = le.fit_transform(data['dept'])

input = data[['satisfaction_level', 'last_evaluation', 'number_project',
              'average_montly_hours', 'time_spend_company', 'Work_accident',
              'promotion_last_5years', 'dept', 'salary']]
outcome = data['left']
# spliting my data into training and testing data training=70%, testing=30%
(input_train, input_test, outcome_train, outcome_test) = train_test_split(input, outcome, test_size=0.3)
# building the model
model = DecisionTreeClassifier()
fitmodel = model.fit(input_train, outcome_train)
predictions = fitmodel.predict(input_test)
# check for the percentage of accuracy of model
print(accuracy_score(outcome_test, predictions))

from sklearn import metrics

print("Precision:", metrics.precision_score(outcome_test, predictions))
# Model Recall
print("Recall:", metrics.recall_score(outcome_test, predictions))
