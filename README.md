# Credit Risk Analysis

**Overview of the analysis**
For this analysis, we wanted to measure the accuracy of some ensembled Machine Learning Models, as well as their behavior when oversampling or undersampling the fit process.

Since Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. We’ll need to employ different techniques to train and evaluate models with unbalanced classes.

<img width="290" alt="balanceTargetValues" src="https://user-images.githubusercontent.com/90414330/157722461-20eb56b6-6b14-4395-a593-4a8f3cab478e.png">

For this challenge, we'll be using `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling. 

**Resources**
- Data Source:
- Software: Visual Studio, Jupyter Notebook, `imbalanced-learn` and `scikit-learn` libraries
- Algorithms: 
	- Oversample the Data. `RandomOverSampler` and `SMOTE`
	- Undersample the data. `ClusterCentroids`
	- Combinatorial Approach of over- and undersampling using the `SMOTEENN` algorithm
	- Machine Learning Models. `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, that reduce bias.

## Results

To describe the accuracy and potential of the machine learning models, we'll be using some metrics as:
- **Precision**: True positives divided by the total number of true predictions.<br>Vice versa, the true negatives divided by the total of negative predictions are called, **Negative Predictive Value**.
- **Recall**: percentage of a certain class correctly identified.
- **Balanced accuracy**: is a metric that one can use when evaluating how good a binary classifier is. It is especially useful when the classes are imbalanced, i.e. one of the two classes appears a lot more often than the other.

**Analysis of each Model**
- Resampling Models to Predict Credit Risk
	- **Naive Random Oversampling.** <br>We have an **accuracy** of 62%, and we can notice that the model misclassifies a lot of High Risk Clients. <br> With the precision obtained from the imbalanced report card, we notice that very few cases of High Risk Models were accurately predicted. <br> <img width="361" alt="naiveRandomOversampling" src="https://user-images.githubusercontent.com/90414330/157724839-0442d22e-4d93-41e2-9f05-945391675757.png"><br><img width="407" alt="naiveRandomOversampling_icr" src="https://user-images.githubusercontent.com/90414330/157724843-b435de4a-2179-4647-af10-6bc369c334a0.png">

	- **SMOTE Oversampling**<br> Similar results to the last model.<br><img width="306" alt="smoteOversampling" src="https://user-images.githubusercontent.com/90414330/157728855-f47a577f-6ddc-47c0-a9dc-e005d4e79279.png"><br><img width="393" alt="smoteOversampling_icr" src="https://user-images.githubusercontent.com/90414330/157728856-8dc08458-8ce2-47d6-9234-72dcb54c8586.png">

	- **ClusterCentroids Undersampling** <br>Notice that the model is still predicting lot of *High Risk Clients*, even more, although the data shows that there aren't many of them.<br><img width="307" alt="clusterCentroidsUndersampling" src="https://user-images.githubusercontent.com/90414330/157730453-3e1672e7-1645-4858-b29a-a3ebb96bc7c9.png"><br><img width="341" alt="clusterCentroidsUndersampling_icr" src="https://user-images.githubusercontent.com/90414330/157730450-46e57a49-5cc0-4a94-be0e-7dd9631b89ac.png">
	- **SMOTEENN for Over and Under Sampling**<br> Here, in contrast of the ClusterCentroids, the model predicts more of low risk clients, but is misclassifing  more of the high risk ones.<br><img width="297" alt="smoteennBothSamplings" src="https://user-images.githubusercontent.com/90414330/157731143-83f5a925-9b90-490c-9429-c93ecb1739a6.png"><br><img width="397" alt="smoteennBothSamplings_icr" src="https://user-images.githubusercontent.com/90414330/157731150-da6874e3-feb4-4a93-9634-0b75f60d98d3.png">
	- **Balanced Random Forest Classifier**<br> In this model, we notice an increase for the accuracy. This because the model reduces weight of the prediction for the *high risk clients*.<br><img width="289" alt="balancedRandomForestClassifier" src="https://user-images.githubusercontent.com/90414330/157732821-2ed7c6dc-8297-4316-a149-50275d522a52.png"><br><img width="341" alt="balancedRandomForestClassifier_icr" src="https://user-images.githubusercontent.com/90414330/157732824-bab29b1b-7613-4743-8c47-97faf10eeaf7.png">
	- **Easy Ensemble AdaBoost Classifier**<br>This model not only predicts fewer *high risk clients*, but also, increases the accuracy for the prediction.<br><img width="312" alt="easyEnsembleAdaBoost" src="https://user-images.githubusercontent.com/90414330/157734313-3cdfa8b3-edb2-43f3-9bf7-afc6018f1901.png"><br><img width="346" alt="easyEnsembleAdaBoost_icr" src="https://user-images.githubusercontent.com/90414330/157734308-a14d89b0-e330-43e6-8829-efdbfca78943.png">

## Summary
Notice that the Easy Ensemble AdaBoost Classifier keeps the weight of the actual data to make the predictions. This helps to reduce the errors and increases the accuracy.

Up to this moment, the model predicts accurately the *Low Risk Clients*, but we can't ensure that we will classify correctly the *High Risk Clients*.
