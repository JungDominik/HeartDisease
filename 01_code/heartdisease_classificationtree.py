# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:38:22 2021

@author: d.jung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay #Instead of plot_confusion_matrix - geht nicht


###Preparation and Data import

path = 'C:\\Pythontest Anaconda\\Statquest\\projects\\HeartDisease'
df = pd.read_csv(path + "\\00_data\\" + "processed.cleveland.data")
df.columns = ["age", "sex","cp", "restbp", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"]

#Identify Missing Data (observations where one of the attributes has blank space / 'na')
# --> columns 'ca' and 'thal' had some observations with the attribute value '?' 
# Show Observations with missing values
# df.loc[(df['ca'] == '?') | (df['thal'] == '?')]
# 6 out of 303 rows
#Note: Categorical Data is given as numbers (eg. 'sex': male = 0, female = 1)

### Data Wrangling

#Missing Values: Needs to be dealt with in one of two ways(Remove or impute/'educated guess') - Classification method cannot have missing value data in the training set
#Chosen here: Remove the observations with missing values
df_nomissing = df.loc[((df['ca'] != '?') & (df['thal'] != '?'))]

#Data Formating

##Split Data into Independent variables (X) and Dependent variable (y)
#Set up Independent Variables df: Make a copy of the given data without the dependent variable 'heart disease'. Use 'real' copying, as opposed to default copy-by-reference
X = df_nomissing.drop('hd', axis = 1).copy()

#Set up Dependent Variables df (heart disease)
y = df_nomissing['hd'].copy()

# Note: Original Data Format as follows: Some variables are float. Remaining variables are categorical --> Encoded as integers 0, 1, 2, 3, ... with respective values
# To utilize categorical variabes in a scikit-learn decision tree, the categorical values need to be transformed so that they are given as specific columns of boolean value  (0 = No /1 = Yes)
# We use One-hot Encoding to do that. Example "Chest Pain" ('cp'). The pandas method is called "get_dummies()" (get dummy columns)
X['cp'].unique() #'Chestpain' variable has only 4 possible values


#Categorical Columns in Dataframe 'X' transformed such that they are binary --> Ready for using  in Decisiontree
X_encoded = pd.get_dummies(X, columns = ['cp', 'restecg', 'slope', 'thal'])
X_encoded.head() # Test if format transform has worked OK


#Transform the dependent variable y: Original is values for likelihood of heart disease on scale 1-5. For simplification of the tree, the values > 1 (= 'chance of heart disease') will be transformed to boolean value 'Heart Disease Yes/No'.
y_index_notZero = y > 0
y[y_index_notZero] = 1
y.unique()


### Tree-building
# Build Preliminary Tree --> Basis for iterative improvement

#Split the data in X and respective dependent variables in y into training and testing sets. Proportion of Training observations <-> Test observations is default value 70 / 30.
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,random_state = 42) #Random seed for the splitting --> Reproducible


clf_dt = DecisionTreeClassifier(random_state = 42) #Initialize Classifier Object
clf_dt.fit(X = X_train, y = y_train) #Fit to the training data


##First round of evaluation
#Visualize for quick evaluation 
plt.figure(figsize = (15,7.5))
plot_tree(clf_dt, filled = True, rounded = True) #Plot the tree
#Evaluation: Tree is too large for meaningful interpretation / possible overfitting. Possibly prune tree

#Check performance with test set data and directly show the results in a confusion matrix. Normalize the confusion matrix (percentage of different groups as opposed to case counts)
plot_confusion_matrix(clf_dt, X_test, y_test)
#Result: 11% False Negative. 8% False Positive


###Improve the tree: Pruning to prevent overfitting
# Method: Pruning based on Cost Complexity. The degree of pruning is defined by parameter 'alpha'. Approach is to find optimal alpha based on evaluating accuracy (confusion matrix) for different values of alpha.


path = clf_dt.cost_complexity_pruning_path(X_train, y_train) #determine values for alpha (minimal cost complexity pruning path)
ccp_alphas = path.ccp_alphas #Extract different values for cost complexity alpha  
ccp_alphas = ccp_alphas[:-1] #Excludes the maximum value for alpha (otherwise the decision tree will be pruned until the root node)

clf_dts = [] #Array to be filled with classification decision trees resulting from the different alpha values

#Loop creates a tree per value of alpha in the pruning path and fills it into the array of trees
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts] #Score: Return mean accuracy for the given train data
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts] #Score: Return mean accuracy for the given test data

fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'Training')
ax.plot(ccp_alphas, test_scores, marker = 'o', label = 'Test')
ax.legend()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Alpha vs Classifier Tree Accuracy for train and test set')
plt.show()

##Evaluation: Rising alpha value represents reduction in overfitting/increase in model bias. Prediction accuracy on the test set is highest (85%) around alpha == 0.01 --> set ccp_alpha = 0.01


###Cross validation to evaluate, whether the previous result holds over different split-ups of the available data into training and test
# cross_val_score(): 1. Split available data up different way, 2. Evaluate the classification tree for this new dataset

clf_dt_forCV = DecisionTreeClassifier(random_state=42, ccp_alpha = 0.016) #New Tree
scores = cross_val_score(clf_dt_forCV,X_train, y_train, cv = 5) #5-fold Cross Validation from the given training set
df_treeAccuracy = pd.DataFrame(data = {'tree':range(5), 'accuracy': scores})

df_treeAccuracy.plot(x = 'tree', y = 'accuracy', marker = 'o') 
#Evaluation: The tree accuracy value is sensitive to the dataset (minimum is ~70%, maximum is 85%)


alpha_loop_values = [] 

#5-Fold Cross validation for each candidate value for alpha
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv = 5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    
    
alpha_results = pd.DataFrame(alpha_loop_values, 
                             columns = ['alpha', 'mean_accuracy', 'std'])

#Plot results as boxplot to choose ideal alpha
alpha_results.plot( x = 'alpha', 
                   y = 'mean_accuracy',
                   yerr = 'std',
                   marker = 'o')

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                                &
                                (alpha_results['alpha'] < 0.019)] ['alpha']
                                
ideal_ccp_alpha = float(ideal_ccp_alpha.unique()) #Result: 0.014190014190014191



### Build Final tree

clf_dt_final = DecisionTreeClassifier(random_state = 42, ccp_alpha = ideal_ccp_alpha)
clf_dt_final.fit(X_train, y_train)

#Evaluate Confusion Matrix for comparison with original tree
plot_confusion_matrix(clf_dt_final, X_test, y_test)


#Draw the improved tree to compare structure
plot_tree(clf_dt_final, filled = True, rounded = True) #Plot the pruned tree
### Evaluation: Large Tree is overfitted. Small tree works better