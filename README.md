# Heart Disease Logistic Regression

## Exercise Summary
Implements logistic regression for heart disease prediction: EDA, training/visualization, regularization, and SageMaker deployment.

## Dataset Description
Kaggle Heart Disease dataset (270 patients; 14 features).
Age range: 29–77 years.  
Cholesterol range: 126–564 mg/dL.  
Heart disease presence rate: 44.4%.

Download source:
`https://www.kaggle.com/datasets/neurocipher/heartdisease`

Local file used:
`src/csv/Heart_Disease_Prediction.csv`

## Notebook Contents
- Step 1: Data loading, binarization, EDA, split, normalization
- Step 2: Logistic regression from scratch (sigmoid, cost, GD)
- Step 3: Decision boundary visualizations for 3 feature pairs
- Step 4: L2 regularization with lambda tuning
- Step 5: SageMaker deployment exploration

## Deployment Evidence
Process summary:
- Trained model and exported weights/bias.
- Created SageMaker notebook/endpoint.
- Deployed real-time endpoint and tested inference.

Endpoint:
`arn:aws:sagemaker:REGION:ACCOUNT:endpoint/ENDPOINT_NAME`

Tested input:
`Age=60, Chol=300, BP=140, Max HR=120, ST depression=1.5, Vessels=2`

Output:
