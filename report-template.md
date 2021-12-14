# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

Using the logistic regression model, we can use the characteristics of loans, such as loan size, interest rate, borrower income etc., to try and predict the outcome of categorical variables, such as if a certain borrower would be a high risk loan or a low risk loan. 

According to the data, there were 75036 low-risk loans and 2500 high risk loans. With such an imbalanced dataset, we might have to try resampling the dataset. I would recommend trying both resampled and non-resampled models. 

Because the nature of this data seems to imply that the variables does not evolve over time, like a time series dataset would, I used the train_test_split function from scikit to split the data randomly into one training section and one testing section for both the predictor variables and the predicted variable. After splitting the data, we fit the model to the training data and use the trained model to predict the testing data.



## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

    Acurracy: 0.9918
    Precision: 0.99
    Recall: 0.99
    F1: 0.99


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
    Accuracy: 0.9938
    Precision: 0.99
    Recall: 0.99
    F1: 0.99

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

I would definitely recommend pursuing different types of machine learning models, just to experiment. The thing is, those description statistics are weighted averages of the score. For example, precision for low-risk in the model is a 1.0, meaning that the times when it predicted low-risk loans, it was always 100% correct. However, precision for high-risk in resampled is only 0.85. But because there is an overwhelming majority of high risk loans made, it averaged out to 0.99 precision. 

Resampling the data doesn't seem to have that much of a change on the performance of the model. In fact, it seems to become a infinitesimal bit worse at accurately predicting high risk loans, which is our primary focus here. 

Although the model have relatively reassuring numbers for a machine learning model, the small fraction that is failures still would eventually translates to a large number of inaccurate predictions if the dataset is large enough. Applied in real life, the amount of money that is still lost from failing to assess high risk loans properly would pile up to be no small sum.