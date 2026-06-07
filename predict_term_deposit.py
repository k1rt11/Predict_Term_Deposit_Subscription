#Swami-Shreeji
import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve
import numpy as np

# load data from CSV file
bank_data = pd.read_csv("bank_full.csv", delimiter=';')

#check for missing values
print("Null values", bank_data.isnull().sum())

#drop rows with missing values
#df_cleaned = df.dropna() #there are no missing values


#columns needed from dataset
relevant_cols = ["job", "marital", "education", "default", "balance", "housing", "loan",
                    "campaign", "pdays", "previous", "poutcome", "y"]

selected_data = bank_data[relevant_cols]

#function to categorise balance
def categorise_balance(balance):
    if balance < 0:
        return 'deficit'
    elif balance == 0:
        return '0'
    elif 1 <= balance < 5000:
        return '1-4999'
    elif 5000 <= balance < 10000:
        return '5000-9999'
    elif 10000 <= balance < 15000:
        return '10000-14999'
    elif 15000 <= balance < 20000:
        return '15000-19999'
    elif 20000 <= balance < 25000:
        return '20000-24999'
    elif 25000 <= balance < 30000:
        return '25000-29999'
    elif 30000 <= balance < 35000:
        return '30000-34999'
    elif 35000 <= balance < 40000:
        return '35000-39999'
    elif 40000 <= balance < 45000:
        return '40000-44999'
    elif 45000 <= balance < 50000:
        return '45000-49999'
    elif 50000 <= balance < 55000:
        return '50000-54999'
    elif 55000 <= balance < 60000:
        return '55000-59999'
    elif 60000 <= balance < 65000:
        return '60000-64999'
    elif 65000 <= balance < 70000:
        return '65000-69999'
    else:
        return '70000-75000'

#function to categorise campaign
def categorise_campaign(campaign):
    if campaign == 1:
        return '1'
    elif 2 <= campaign <= 5:
        return '2-5'
    elif 6 <= campaign <= 10:
        return '6-10'
    elif 11 <= campaign <= 15:
        return '11-15'
    elif 16 <= campaign <= 20:
        return '16-20'
    else:
        return 'greater than 20'

#function to categorise pdays
def categorise_pdays(pdays):
    if pdays == -1:
        return 'not contacted'
    elif -1 < pdays <= 220:
        return 'recently contacted'
    elif 220 < pdays < 440:
        return 'contacted some time ago'
    elif 440 < pdays <= 660:
        return 'contacted long time ago'
    else:
        return 'contacted very long time ago'
   
#function to categorise previous
def categorise_previous(previous):
    if previous == 0:
        return 'no previous contact'
    elif 0 < previous <= 5:
        return '1-5 previous contacts'
    elif 5 < previous <= 10:
        return '6-10 previous contacts'
    elif 10 < previous <= 15:
        return '11-15 previous contacts'
    elif 15 < previous <= 20:
        return '16-20 previous contacts'
    else:
        return '21-25 previous contacts'
   


# create new column for categorised features
selected_data["balance_range"] = selected_data["balance"].apply(categorise_balance)
selected_data["num_contacts_campaign"] = selected_data["campaign"].apply(categorise_campaign)
selected_data["past_contact"] = selected_data["pdays"].apply(categorise_pdays)
selected_data["prev_campaign_contacts"] = selected_data["previous"].apply(categorise_previous)

#drop old columns
selected_data.drop(columns=["balance", "campaign", "pdays", "previous"], inplace=True)


# function for latent variable -financial stability
def categorise_financial_stability(row):
    if (row["balance_range"] == '0' or row["balance_range"] == 'deficit') and row["job"] == 'unemployed':
        return 'unstable'
    else:
        return 'stable'

selected_data['financial_stability'] = selected_data.apply(categorise_financial_stability, axis=1)

#rename columns for clarity
selected_data.rename(columns = {'housing':'housing_loan'}, inplace = True)
selected_data.rename(columns = {'default':'default_credit'}, inplace = True)
selected_data.rename(columns = {'poutcome':'prev_campaign_outcome'}, inplace = True)
selected_data.rename(columns = {'y':'subscription'}, inplace = True)


#splitting the data into the training set and the test set
train, test = train_test_split(selected_data, test_size=0.3, random_state=42)

#structure learning the model
hc = HillClimbSearch(train)

# using the Hill-Climbing algorithm with BIC score (scoring method) to learn structure
best_model = hc.estimate(scoring_method=BicScore(train))

print("Best model edges after hill climbing: ", best_model.edges())

learned_model = nx.DiGraph()

learned_edges = best_model.edges()

learned_model.add_edges_from(learned_edges)

plt.figure(figsize=(10, 8))
nx.draw_circular(learned_model, with_labels=True, node_size=9000, font_size=15, font_weight='bold',
         arrowsize = 20)  

plt.show()


#removing the marital node and any edges coming to it- no edges going from it
best_model.remove_node('marital')

#removing default node and edge coming to it- no edges going from it
best_model.remove_node('default_credit')

#removing campaigns node and edge coming to it- no edges going from it
best_model.remove_node('num_contacts_campaign')

#removing prev_contacts node and edge coming to it- no edges going from it
best_model.remove_node('prev_campaign_contacts')

#removing unnecessary edges
best_model.remove_edge('housing_loan', 'past_contact')
best_model.remove_edge('housing_loan', 'loan')
#best_model.remove_edge('job', 'housing_loan')

print("Best model edges after removing unnecessary dependencies: ", best_model.edges())

bayesian_network = BayesianNetwork(best_model)

#learning th eparameters using the Bayesian Dirichlet prior
bayesian_network.cpds = []

#tested different values for sample size and found 10 to be the most optimal for the best accuracy
bayesian_network.fit(data=train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10) #Bayesian Dirichlet equivalent uniform prior
 
#drawing Bayesian network
bank_marketing = nx.DiGraph()

edges = best_model.edges()

bank_marketing.add_edges_from(edges)

# position nodes using spring layout so that nodes do not overlap each other
pos = nx.spring_layout(bank_marketing, seed=42)

# manually specify x- and y-coordinates positions for specific nodes
pos['education'] = (pos['education'][0] + 0.7, pos['education'][1] + 0.4)
pos['loan'] = (pos['loan'][0] - 1 , pos['loan'][1] - 0.3)
pos['job'] = (pos['job'][0] +0.15, pos['job'][1])
pos['past_contact'] = (pos['past_contact'][0], pos['past_contact'][1] + 0.5)
pos['balance_range'] = (pos['balance_range'][0] - 0.5 , pos['balance_range'][1])
pos['housing_loan'] = (pos['housing_loan'][0] , pos['housing_loan'][1] - 0.2)
pos['subscription'] = (pos['subscription'][0] -0.1, pos['subscription'][1] - 1.3)
pos['financial_stability'] = (pos['financial_stability'][0] +0.1 , pos['financial_stability'][1] + 0.2)


plt.figure(figsize=(10, 8))
nx.draw(bank_marketing, pos, with_labels=True, node_size=14000, font_size=15, font_weight='bold',
         arrowsize = 20, connectionstyle="arc3,rad=0.1")  

plt.show()



train.drop(columns=["marital", "default_credit", "num_contacts_campaign", "prev_campaign_contacts"], inplace=True)

# Initialize VariableElimination with the trained Bayesian Network
update_financial_stability = VariableElimination(bayesian_network)



# Expectation-Maximizationalgorithm to estimate latent variable 'financial_stability'
def expectation_maximization(bayesian_network, train):
    # initialize parameter for convergence check
   
    prev_params = None
    #will take some time
    tol = 1e-3

    while True:
        # find posterior probabilities of 'financial_stability' given observed data
        posterior_stability = []
        for index, row in train.iterrows():
            evidence = row.drop('financial_stability').to_dict()  
            query_result = update_financial_stability.query(variables=['financial_stability'], evidence=evidence)
            posterior_stability.append(query_result.values)

        # updating parameters using expected values
        bayesian_network.fit(data=train, estimator=BayesianEstimator, prior_type="Bdeu", equivalent_sample_size=10,
                            state_names=train['financial_stability'].unique().tolist())
       
        curr_params = [cpd.values.flatten() for cpd in bayesian_network.get_cpds()]

        #convergence check
        if prev_params is not None:
            # make sure all arrays consistent shapes by reshaping them
            max_shape = max(param.shape for param in prev_params)
            prev_params = [np.pad(param, [(0, max_shape[0] - param.shape[0])], mode='constant') for param in prev_params]
            curr_params = [np.pad(param, [(0, max_shape[0] - param.shape[0])], mode='constant') for param in curr_params]

            # converting lists of arrays to numpy arrays for element-wise subtraction
            prev_params = np.array(prev_params)
            curr_params = np.array(curr_params)

            # absolute difference between parameters
            param_diff = np.abs(prev_params - curr_params)

            # checking if maximum difference is less than tolerance
            if np.max(param_diff) < tol:
                break  

        # updating previous parameters for next iteration
        prev_params = curr_params

    return bayesian_network


# EM to estimate financial stability
bayesian_network_em = expectation_maximization(bayesian_network, train)
#tested different values for sample size and found 10 to be the most optimal for the best accuracy
bayesian_network_em.fit(data=train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10) #Bayesian Dirichlet equivalent uniform prior

print(f'Check model: {bayesian_network.check_model()}\n')
for cpd in bayesian_network.get_cpds():
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')

# Local independencies
for x in bayesian_network.nodes():
  print(" Independencies :",x,'\n',bayesian_network.local_independencies(x))
  print(" Descendants :",x,'\n',bayesian_network.get_children(x))
  print('\n')

#inference of variables
term_deposit_subscription = VariableElimination(bayesian_network_em)

test.drop(columns=["marital", "default_credit", "num_contacts_campaign", "prev_campaign_contacts"], inplace=True)

X_test = test.drop(columns=['subscription'])  # features
y_test = test['subscription']  #target variable

# Initialize VariableElimination with the trained Bayesian Network
term_deposit_subscription = VariableElimination(bayesian_network)

# Perform inference on the test data
predicted_y = []
for index, row in X_test.iterrows():
    evidence = row.to_dict()
    
    # Query the probability distribution of 'y' given evidence from test data
    query_result = term_deposit_subscription.query(variables=['subscription'], evidence=evidence)
    
    # Get the most probable state of 'y' (prediction)
    predicted_state = query_result.values.argmax()
    predicted_y.append(predicted_state)

# Convert predicted_y to a pandas Series
predicted_y = pd.Series(predicted_y, index=test.index)

# Extract true y values from the test data
true_y = test['subscription']
true_y_numeric = true_y.replace({'no': 0, 'yes': 1})

# evaluation metrics
accuracy = accuracy_score(y_true=true_y_numeric, y_pred=predicted_y)
confusion = confusion_matrix(y_true=true_y_numeric, y_pred=predicted_y)
classification_report = classification_report(y_true=true_y_numeric, y_pred=predicted_y)
# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_y_numeric, predicted_y)
roc_auc = roc_auc_score(true_y_numeric, predicted_y)

# precision-recall curve
precision, recall, _ = precision_recall_curve(true_y_numeric, predicted_y)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report)
print("ROC AUC:", roc_auc)

# precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(y_true=true_y_numeric, y_score=predicted_y)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calibration Curve
true_prob, pred_prob = calibration_curve(y_true=true_y_numeric, y_prob=predicted_y, n_bins=10)
plt.plot(pred_prob, true_prob, marker='.')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.show()
