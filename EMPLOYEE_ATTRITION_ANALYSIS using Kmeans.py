#import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs

excelfile = pd.ExcelFile('real.xlsx')
data = excelfile.parse('Sheet1')
print(data.head())
print(data.tail())
print(data.describe())
print(data.left.value_counts())

left = data.groupby('left')
print(left.mean())

num_projects=data.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Number-of-Projects')
plt.ylabel('Number-of-Employees')
plt.savefig('pic1.png')
#plt.show()

time_spent=data.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
plt.xlabel('Number-of-Years-Spend-in-Company')
plt.ylabel('Number of Employees')
plt.savefig('pic2.png')
#plt.show()

features=['number_project','time_spend_company', 'promotion_last_5years','salary']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data)
    plt.xticks(rotation=90)
    plt.title("No. of employee")


features = ['number_project', 'time_spend_company', 'promotion_last_5years', 'salary']
fig = plt.subplots(figsize=(10, 15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=data, hue='left')
    plt.xticks(rotation=90)
    plt.title("No. of employee")
    plt.show()


#cluster analysis
#import module
from sklearn.cluster import KMeans
# Filter data
left_emp =  data[['satisfaction_level', 'last_evaluation']][data.left == 0]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_emp)
left_emp['label'] = kmeans.labels_
fig = plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'],cmap='Accent')
plt.xlabel('Satisfaction_Level')
plt.ylabel('Last_Evaluation')
plt.show()


#BUILDING OUR MODEL
from sklearn import metrics
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
labelE = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=labelE.fit_transform(data['salary'])
data['dept']=labelE.fit_transform(data['dept'])
#Spliting the Features data
X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company',
       'promotion_last_5years','salary']]
y=data['left']

# Import train test split function
from sklearn.model_selection import train_test_split

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
#Train the model using the training sets
gb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gb.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test, y_pred))
print("precision:",metrics.precision_score(y_test, y_pred))
print("recall:",metrics.recall_score(y_test, y_pred))
