import pandas as pd
import sklearn.model_selection as skm
import sklearn.metrics as skmet
import sklearn.svm as sksvm
from sklearn.neighbors import KNeighborsClassifier

VAX = pd.read_csv('/Users/austingarrett/Desktop/MSBA/Fall 2021/542 Data Mining/542 Project Files/VACCINE_DATA.csv',
                  delimiter=',')

VAX.dtypes

# SVM Model

# Dropping non-numeric variable for testing purposes
VAX = VAX.drop(columns=['State', '2020', 'E.Hesitant', 'FIPS'], axis=1)
VAX['Hesitancy.disc'].value_counts()

# Scale the data

# Remove the classifier column
X = VAX.drop(['Hesitancy.disc'], axis='columns')
X.head()
X.info()


# Function to apply the min-max scaling in Pandas using the .min() and .max() methods
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
    return df_norm


# Call the min_max_scaling function
VAX_normalized = min_max_scaling(X)

# Response variable is the hesitancy.disc (1,2,3,4)
y = VAX['Hesitancy.disc']
print(y)

# Initial Model (Don't need to run)
# Split into training and testing data, testing is 20% of original data
X_train, X_test, y_train, y_test = skm.train_test_split(VAX_normalized, y, test_size=.2)
len(X_train)
len(X_test)

# Fit SVC model classifying no change, increase, or decrease
model = sksvm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy of model without tuning
model.score(X_test, y_test)

# Tuning Model
# Apply kernels to transform the data to a higher dimension

kernels = ['RBF', 'Linear']


def getClassifier(ktype):
    if ktype == 0:
        # rbf kernel
        return sksvm.SVC(kernel='rbf', gamma="auto")
    elif ktype == 1:
        # Linear kernel
        return sksvm.SVC(kernel='linear', gamma="auto")


# Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
param_grid = {'C': [150], 'gamma': [0.5], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
# Initial run's best predictors: C=10, gamma=0.1
# Second run's best predictors: SVC(C=100, gamma=0.2)
# Third run's best predictors: SVC(C=100, gamma=0.5)
# Fourth run's best predictors: SVC(C=150, gamma=0.35) idk why I lowered the gamma's below 0.5 lol
# Fifth run's best predictors: SVC(C=150, gamma=0.5) Weighted Avg Acc of 0.72 (Taking out FIPS drops acc to 0.7)


# Create a GridSearchCV object and fit it to the training data
grid = skm.GridSearchCV(sksvm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Find the optimal parameters
print(grid.best_estimator_)

# Prediction Matrix
grid_predictions = grid.predict(X_test)
print(skmet.confusion_matrix(y_test, grid_predictions))
print(skmet.classification_report(y_test, grid_predictions))



# KNN Model

# Initial model no need to run this
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",skmet.accuracy_score(y_test, y_pred))


# Grid search
param_grid = {'n_neighbors': [3, 5, 9, 11],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

grid = skm.GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print("Accuracy:", skmet.accuracy_score(grid_predictions, y_pred))
