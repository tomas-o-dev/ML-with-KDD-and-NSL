
These notebooks use two "toy" datasets and a local library
to illustrate aspects of the classification ML Pipeline

The "bank churn" dataset has text and numeric features, 
and is a two-class (binary) prediction problem

The "tomato juice" dataset has only numeric features, 
and is an imbalanced multi-class prediction problem

DataPrep.ipynb  (churn)
	dataset preparation and model results

EvalViz.ipynb  (toju)
	visualising the confusion matrix and model results

FeatSelCorr.ipynb  (churn)
	feature correlations with target labels
    ("Ground Truth") 

FeatSelCV.ipynb  (churn)
	feature selection based on their influence on the 
	model results - mlxtend functions (efficient)

hypopt-KNN.ipynb  (toju) 
	hyperparameter optimisation using Grid Search and 
	similar techniques for individual hyperparameters

imBalance.ipynb  (toju)
	Over/Under sampling to address imbalanced classes 

mComp-CV.ipynb  (toju) and
mComp-ptst.ipynb  (toju)
	model comparison using cross validation and 
	statistical hypothesis testing



>=< - Notes - >=< 
    Extended evaluation metrics


[1] Bias // Variance decomposition of loss function

Most machine learning algorithms use loss function in the process of 
optimization, or finding the best parameters (weights) for your data. 
A "loss function" measures how far an estimated value is from its true 
value. The "loss" is a number indicating how bad the model's prediction 
was on a single example. If the prediction is perfect, the loss is zero; 
otherwise, the loss is greater.

In supervised learning, a machine learning algorithm builds a model by 
examining many examples and attempting to find a model that minimizes 
loss for the whole dataset; this process is called "empirical risk 
minimization".

The core idea is that we cannot know exactly how well an algorithm 
will work in practice (the "true risk") because we don't know the true 
distribution of data that the algorithm will work on, but we can measure 
its performance on a known set of training data (the "empirical risk") 
by averaging the loss function on the training set to see the 
"expected loss".

In mathematical terms, the "excess risk" of the output of the learning 
algorithm can be decomposed as "approximation error plus estimation error". 
In statistical terms, we can decompose a loss function into three terms: 
bias, variance, and a noise term that we can ignore for simplicity.

In general, we can say that a model with high bias will be more consistent 
with predictions on different test sets - which could be consistently good  
or consistently bad. A model with high variance will give accurate 
predictions for some test sets and poor predictions for others. 

Given test/train data and labels, the bias_variance_decomp() function from 
mlxtend.evaluate breaks down the performance of an estimator into bias and 
variance. It also returns the expected loss, which, subtracted from one, 
gives us a measure of "goodness". Together these give us insight into how 
accurate and consistent the predictions will be with new test data. 

Currently supported loss functions for the bias-variance decomposition are 
'0-1_loss' for classification and 'mse' (mean squared error) for regression.
There is a note in the code about the requirement for numpy ndarrays and 
numeric labels.



[2] Balanced accuracy for multi-class problems

Accuracy = tp+tn / (tp+tn+fp+fn) doesn't work well for unbalanced classes. 

There are alternative definitions of "Balanced Accuracy" to deal with 
imbalanced datasets and binary and multiclass classification problems.

One definition of Balanced Accuracy = TPR+TNR/2
TPR = true positive rate = tp/(tp+fn) = sensitivity = Recall
TNR = true negative rate = tn/(tn+fp) = specificity
This method gives almost the same results as ROC AUC Score 
for a two-class problem.

In a multi-class setting, we can generalize the computation of the accuracy 
as the average of the per-class predictions in the confusion matrix.

The balanced_accuracy_score() from sklearn.metrics is useful when 
a scorer() can be specified as a parameter (multi-class cross-validation 
for example). By default returns average recall for each class, equivalent 
to the weighted_accuracy_scorein the classification report. When the 
"adjusted" parameter (default=False) is True, the result is adjusted for 
chance, so random performance score is 0 and perfect performance score is 1.
 
The accuracy_score() from mlxtend.evaluate is designed to calculate  
three different ways, depending on the method= parameter:
# method : 'standard' - overall accuracy (default)
#          'binary'   - accuracy for a single class
#          'average'  - average of all single-class accuracy scores

The "binary" or "per-class" accuracy is the stamdard accuracy of one 
class versus all remaining datapoints in the dataset. To compute the 
average per-class accuracy, the binary accuracy for each class label 
is first computed separately; for example, following the diagonal of 
a 3-class confusion matrix, when class 1 is the positive class, 
class 0 and 2 together are the negative class.

see also:
https://scikit-learn.org/stable/modules/model_evaluation.html
https://imbalanced-learn.org/stable/user_guide.html

