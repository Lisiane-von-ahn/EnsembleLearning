# EnsembleLearning
Project Title
Ensemble Learning for Bank Marketing Campaigns

Description
This project explores various machine learning techniques to predict the success of bank marketing campaigns. We experiment with decision trees, random forests, and XGBoost to evaluate their performance in terms of F1 score, using balanced class weights to handle imbalanced data. The final goal is to identify the best model for predicting customer responses.

Table of Contents
Installation
Data
Models
Decision Tree
Random Forest
XGBoost
Performance Evaluation
Conclusion
To Do
References
Installation
To run the code in this repository, you will need to have Python installed along with the following packages:

numpy
pandas
scikit-learn
matplotlib
xgboost
You can install these packages using pip:

bash
Copier le code
pip install numpy pandas scikit-learn matplotlib xgboost
Data
The dataset used in this project is the Bank Marketing dataset. It contains various features such as age, job, marital status, and more, along with the target variable indicating whether a client has subscribed to a term deposit.

Models
Decision Tree
We first train a Decision Tree classifier, tuning its max_depth and balancing the classes using the class_weight='balanced' parameter.

Random Forest
Next, we train a Random Forest classifier. We use GridSearchCV to find the optimal parameters for n_estimators and max_depth.

XGBoost
Lastly, we implement an XGBoost classifier, also using GridSearchCV to optimize n_estimators, max_depth, and learning_rate. The sample_weight parameter is used to handle class imbalance.

Performance Evaluation
We evaluate the models using F1 score as our primary metric, due to the imbalanced nature of the dataset.

python
Copier le code
# Example performance evaluation code for a decision tree
from sklearn.metrics import f1_score

# Decision Tree on test set
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=5, class_weight='balanced')
decision_tree.fit(X_train_transformed, y_train)

y_pred_test = decision_tree.predict(X_test_transformed)
f1_test = f1_score(y_test, y_pred_test)

print("F1 test (Decision Tree): ", f1_test)
Conclusion
After evaluating the models, we noticed the following:

Decision Tree: Provided a baseline performance.
Random Forest: Showed improvement over the Decision Tree.
XGBoost: Achieved the best performance in terms of F1 score.
Based on the results, we recommend using the XGBoost model for predicting the success of bank marketing campaigns due to its superior performance.

To Do
Explore more advanced balancing techniques.
Test additional boosting algorithms.
Implement a customized bagging classifier.
Visualize model performance with more plots.
References
Scikit-Learn Documentation
XGBoost Documentation
UCI Machine Learning Repository: Bank Marketing Dataset
https://medium.com/@riteshgupta.ai/accuracy-precision-recall-f-1-score-confusion-matrix-and-auc-roc-1471e9269b7d
https://medium.com/@silvaan/ensemble-methods-tuning-a-xgboost-model-with-scikit-learn-54ff669f988a
https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=Accuracy%20shows%20how%20often%20a,when%20choosing%20the%20suitable%20metric.
