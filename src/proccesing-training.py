import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier  # Example model, replace with your model
import joblib

data = pd.read_csv("BankChurners.csv")

# Handle missing values for numerical columns
numerical_columns = ['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy']
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Handle missing values for categorical columns
categorical_columns = ['Education_Level', 'Marital_Status', 'Income_Category']
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], drop_first=True)

# Map Gender and Attrition_Flag to numerical values
data_encoded['Gender'] = data_encoded['Gender'].map({'M': 0, 'F': 1})
data_encoded['Attrition_Flag'] = data_encoded['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Drop irrelevant columns
columns_to_drop = [
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
data_encoded.drop(columns_to_drop, axis=1, inplace=True)

selected_features = [
    'Total_Trans_Ct', 'Total_Trans_Amt', 'Customer_Age', 'Months_on_book',
    'Avg_Utilization_Ratio', 'Total_Revolving_Bal', 'Gender',
    'Income_Category_Less than $40K', 'Card_Category_Silver',
    'Income_Category_$80K - $120K', 'Card_Category_Gold', 'Credit_Limit'
]

data_encoded.to_csv("cleaned_data.csv", index=False)  # Save cleaned data without scaling

#Training our model + Saving
# Create the feature matrix (X) and target variable (y)
X = data_encoded[selected_features]
y = data_encoded['Attrition_Flag']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection (Random Forest Classifier)
rf_model = RandomForestClassifier(random_state=42)  # You can specify hyperparameters here if needed

# Training the Model
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report Test:\n", report)


# Define a smaller set of hyperparameters and their possible values
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the grid search on your training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Model Evaluation with the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Best Model: {accuracy:.2f}")

# Generate a classification report for the best model
report = classification_report(y_test, y_pred)
print("Classification Report for Best Model:\n", report)


# Define your model
model = RandomForestClassifier(random_state=42)

# Define the number of folds and the type of cross-validation
num_folds = 5  # You can choose the number of folds you want
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation and get accuracy scores
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Print the accuracy scores for each fold
for fold, score in enumerate(scores, start=1):
    print(f'Fold {fold}: Accuracy = {score:.2f}')

# Calculate the mean and standard deviation of the accuracy scores
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print(f'Mean Accuracy = {mean_accuracy:.2f}, Standard Deviation = {std_accuracy:.2f}')

# Assuming you have a trained model named 'best_model'
model_filename = 'best_model.pkl'  # Choose a filename and extension (commonly .pkl)
joblib.dump(best_model, model_filename)