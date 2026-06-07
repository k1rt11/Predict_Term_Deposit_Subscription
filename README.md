# Predicting Bank Term Deposit Subscriptions with Bayesian Networks

## Overview

This project uses a Bayesian Network to predict whether a bank customer will subscribe to a term deposit using the UCI Bank Marketing dataset.

The model includes data preprocessing, structure learning, Bayesian parameter estimation, and exact inference through Variable Elimination.

## Results

- Accuracy: 89.27%
- ROC-AUC: 0.586

The model achieved strong overall accuracy and demonstrates the application of probabilistic graphical models for customer behaviour prediction.

## Techniques Used

- Bayesian Networks
- Hill Climbing Structure Learning
- Bayesian Information Criterion (BIC)
- Bayesian Parameter Estimation (BDeu Prior)
- Variable Elimination
- Python, Pandas, Scikit-learn, pgmpy, NetworkX

## Dataset

Bank Marketing Dataset (UCI Machine Learning Repository):

https://archive.ics.uci.edu/dataset/222/bank+marketing

Place `bank_full.csv` in the project directory before running the script.

## Running the Project

```bash
pip install -r requirements.txt
python predict_term_deposit.py
