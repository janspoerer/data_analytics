###################################################################
# Part 0: Sample reduction variable, library imports, data import #
########################################################################

# Sample scaling (downsampling for development purposes)
sample_size_ratio_config = 1 # sample_size = 1 means that all data will be used.
test_run = 0 # Set to 1 to just check for syntax and run with very small sample

# Imports except for model-specific sklearn imports
# Main 3 libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# Additional libraries
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D

# Data import
df = pd.read_csv("Intelligent_Couponing.csv")
if test_run == 1:
    rows = np.random.choice(df.index.values, 1000)
    df = df.ix[rows]
if sample_size_ratio_config != 1:
    import math
    df = df.sample(n=math.floor(sample_size_ratio_config*len(df.count())))
number_observations_original = df.customernumber.count()


############################
# Part 1: Business problem #
############################

## Why is dependent variable important to know?
### The dependent variable indicates if a given customer will re-order within 90 days without getting a voucher.
### It is relevant because if a customer is likely to re-order within 90 days without getting a voucher, it may not make economic sense to provide him with a voucher.
### The voucher is worth 5 EUR.
### To evaluate the imporance of this order (i.e., the contribution margin), one also needs to be able to approximate the expected contribution margin for the next order.
### In addition to the contribution margin of the next order, the change in customer lifetime value is imporant to estimate.
## Which aspects have to be considered?
### The customer lifetime value in case of providing the customer with a voucher vs. not providing the customer with a voucher is the main concern. The customer lifetime value is approximated by the expected past contribution margin and by the purchasing behavior of the customer. One deciding factor could be the return rate of the customer. Customers with high return rates should not be actively targeted because we believe that their contribution margin is lower than the contribution margin of customers that do not return their orders.
### Here, do not directly calculate the customer lifetime value. We also don't have enough information to calculate the expected contribution margin over the next 90 days. Therefore, we simply assume that we want to provide vouchers to customers that (1) are not likely to order otherwise, (2) and that are expected to place orders that are profitable.



###################################
# Part 2: Variable adaption & EDA #
###################################

print(df.shape)
print(df.dtypes)
print(list(df))
print(df.isnull().sum())
print(df['target90'].value_counts())

df = df.drop(['customernumber'], axis=1)

# If deliverytype is collection/pickup, the deliverypostcode is always NAN. We think the postcodes may have some predictive power as a categorical variable, but they would require too much computational power because we would have to create many dummy variables. Therefore, we drop the invoicepostcode and the deliverypostcode and only leave the deliverytype.
df = df.drop(['delivpostcode', 'invoicepostcode'], axis=1)

# Many missing values here and many different codes. We think the fact that a customer came as a result of a campaign is important. Therefore, the column is transformed to a dummy variable in the next line, we do not care about the specific type of advertising campaign.
print("Unique advertising data codes: ", df.advertisingdatacode.unique()) # Too many different codes, not usable as a categorical variable in that form.
df['hasadvertisingdatacode_custom'] = np.where(df['advertisingdatacode'].isna(), 0, 1) # Better transform it into boolean; it is not important which campaign it was, it is more important that the customer was acquired through a campaign at all.
df = df.drop(['advertisingdatacode'], axis=1) # And delete the old column.
print(df['hasadvertisingdatacode_custom'].value_counts())

# We assume the people that have "other" domains may take their account more serious (because they do not just use a trash email account) or they may be more sophisticated users.
df['domain_custom'] = np.where(df['domain'] == 12, 0, 1)

# Already covered by number of individual item types (w0, w1...).
df = df.drop(['numberitems'], axis=1)

# Only 6 orders from that category shipped -> Drop.
df = df.drop(['w8'], axis=1)

# Change data type of date
df['date'] = pd.to_datetime(df['date'])
df['datecreated'] = pd.to_datetime(df['datecreated'])
df['deliverydatepromised'] = pd.to_datetime(df['deliverydatepromised'], errors='coerce')
df['deliverydatereal'] = pd.to_datetime(df['deliverydatereal'], errors='coerce')
# Assumption: Customers are dissatisfied when there is a delay and the duration of the delay is not very important.
df.loc[(df['deliverydatereal'] > df['deliverydatepromised']), 'deliveryontime_custom'] = 0
print("custom", df['deliveryontime_custom'].isna().sum())
df.loc[(df['deliverydatereal'] <= df['deliverydatepromised']), 'deliveryontime_custom'] = 1
print("custom", df['deliveryontime_custom'].isna().sum()) # Intangible good, no delivery time
df.loc[(df['w3'] > 0), 'deliveryontime_custom'] = 1
print("custom", df['deliveryontime_custom'].isna().sum()) # Intangible good, no delivery time
df.loc[(df['w5'] > 0), 'deliveryontime_custom'] = 1
print("custom", df['deliveryontime_custom'].isna().sum()) # Intangible good, no delivery time
df.loc[(df['w10'] > 0), 'deliveryontime_custom'] = 1
print("custom", df['deliveryontime_custom'].isna().sum()) # Intangible good, no delivery time
df.loc[(df['cancel'] > 0), 'deliveryontime_custom'] = 1
print("custom", df['deliveryontime_custom'].isna().sum())
df = df.dropna(subset=['deliveryontime_custom']) # Only 34 NAs left for deliveryontime_custom. We just drop those.
df = df.drop(['deliverydatepromised', 'deliverydatereal'], axis=1) # Not needed anymore.

# Create new dummy column that is 1 if the customer ordered at the day of account creation.
df['time_to_first_order_custom'] = df['date'] - df['datecreated']
df['time_to_first_order_custom'] = df['time_to_first_order_custom'].dt.days
print(df['time_to_first_order_custom'].dtypes)
df['time_to_first_order_custom'] = pd.to_numeric(df['time_to_first_order_custom'])
df['ordered_directly_after_account_creation_custom'] = np.where(df['time_to_first_order_custom'] == 0, 1, 0)
df = df.drop(['time_to_first_order_custom', 'date', 'datecreated'], axis=1) # Not needed anymore.

# We used our domain knowledge to decide against performing PCA to reduce the number of features. We believe the variables selected are meaningful and we already identified colinear variables (number of items vs. number of individual item types; weight vs. number of individual item types).

# We are more interested if there were any used items in the order and less interested in the exact number of used items.
df['used_custom'] = np.where(df['used'] >= 0, 1, 0)
df = df.drop(['used'], axis=1)

# We are more interested if there were any canceled items in the order and less interested in the exact number of canceled items.
df['cancel_custom'] = np.where(df['cancel'] >= 0, 1, 0)
df = df.drop(['cancel'], axis=1)

# EPA showed that there is little correlation between remit and the target variable.
df = df.drop(['remi'], axis=1)

# Printing unique values of categorical variables to detect outliers.
print("Columns with missing values: ", df.columns[df.isnull().any()])
print("Unique salutations: ", df.salutation.unique()) # No outliers found (0  = Ms; 1 = Mr; 2 = Company)
print("Values for each category: ", df.salutation.value_counts(normalize=True))
print("Unique titles: ", df.title.unique()) # No outliers found (0 = no title; 1 = title)
print("Values for each category: ", df.title.value_counts(normalize=True))
df = df.drop(['title'], axis=1) # Drop because very rare (<0.01%) (but still a burden when computing).
print("Unique domains: ", df.domain.unique()) # No outliers found (0 = aol.com; 1 = arcor.de; 2 = freenet.de; 3 = gmail.com; 4 = gmx.de; 5 = hotmail.de; 6 = online.de; 7 = onlinehome.de; 8 = t-online.de; 9 = web.de; 10 = yahoo.com; 11 = yahoo.de; 12 = others)
print("Values for each category: ", df.domain.value_counts(normalize=True))
print("Unique newsletters: ", df.newsletter.unique()) # No outliers found (0 = no; 1 = yes)
print("Values for each category: ", df.newsletter.value_counts(normalize=True))
print("Unique models: ", df.model.unique()) # No outliers found (models 1, 2, 3)
print("Values for each category: ", df.model.value_counts(normalize=True))
df = df.drop(['model'], axis=1) # We did not understand what model means and thought it is better to drop it. We cannot test it for plausibility.
print("Unique payment types: ", df.paymenttype.unique()) # No outliers found (0 = Payment on invoice; 1 = Cash payment; 2 = Transfer from current account; 3 = Transfer from credit card)
print("Values for each category: ", df.paymenttype.value_counts(normalize=True))
print("Unique delivery types: ", df.deliverytype.unique()) # No outliers found (Delivery type: 0 = Dispatch; 1 = collection)
print("Values for each category: ", df.deliverytype.value_counts(normalize=True))
print("Unique voucher redeptions: ", df.voucher.unique()) # No outliers found (Voucher redeemed: 0 = No; 1 = Yes)
print("Values for each category: ", df.voucher.value_counts(normalize=True))
print("Unique values of goods (case): ", df.case.unique()) # No outliers found (Value of goods: 1 = low; 5 = high)
print("Values for each category: ", df.case.value_counts(normalize=True))
print("Unique gift options: ", df.gift.unique()) # Almost never with gift option, drop (Gift option: 0 = No; 1 = Yes)
print("Values for each category: ", df.gift.value_counts(normalize=True))
df = df.drop(['gift'], axis=1)
print("Unique entry: ", df.entry.unique()) # No outliers found (Entry into the shop: 0 = Shop; 1 = Partner)
print("Values for each category: ", df.entry.value_counts(normalize=True))
print("Unique points redeemed: ", df.points.unique()) # Always 0, drop (Points redeemed: 0 = No; 1 = Yes)
print("Values for each category: ", df.points.value_counts(normalize=True))
df = df.drop(['points'], axis=1)
print("Unique shipping costs incurred: ", df.shippingcosts.unique()) # No outliers found (Shipping costs incurred: 0 = No; 1 = Yes)
print("Values for each category: ", df.shippingcosts.value_counts(normalize=True))

# Printing ranges of numerical variables to detect outliers
print(df.describe())
# Some weights are 0, but this is plausible. There are non-tangible items in the product portfolio of the company. In fact, we checked the orders that have a weight of 0 and they consist of non-tangible items.
# The weight has a strong correlation with the number of items and the types of items. As we still have these variables in our dataframe, we drop weight.
df = df.drop(['weight'], axis=1)

# Number of observations in original dataset:
print("Number of observations in original dataset: ", number_observations_original)
# Number of observations after deleting outliers:
number_observations_after = df.count()
print("Number of observations after deleting outliers: ", number_observations_after)

# Creating dummies and dropping original category columns.
df = pd.get_dummies(df, columns=['salutation', 'domain_custom', 'newsletter', 'paymenttype', 'deliverytype'])
df = pd.get_dummies(df, columns=['voucher', 'case', 'entry', 'cancel_custom', 'used_custom', 'shippingcosts', 'hasadvertisingdatacode_custom'])

# Standardize and normalize
df_standardized = (df[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']]-df[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']].mean())/df[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']].std()
df_normalized = (df_standardized[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']]-df_standardized[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']].min())/(df_standardized[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']].max()-df_standardized[['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w9', 'w10']].min())
df.merge(df_normalized)

print(df.shape)
# Final deletion of NAs
df = df.dropna(how='any') # Little number of NAs was found! Data is already pretty clean.
print(df.shape)



###############################
# Part 3: Stratified sampling #
###############################

# Balancing (to minimize getting biased samplies during sampling)
# Since we have only about 6000 rows, we do not need to down sample as he recommended
print(df['target90'].value_counts())

df_majority = df[df.target90==0]
df_minority = df[df.target90==1]
df_majority_balanced = resample(df_majority,
                                n_samples=len(df_minority),
                                replace=False, random_state=0) # random_state will fix the output of the "random" sample. Delete the parameter to make the sample truly random.
df_balanced = pd.concat([df_majority_balanced,
                                  df_minority])
print(df_balanced['target90'].value_counts())

# Partition into training and test
X = df.drop('target90', axis = 1)
Y = df['target90']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, #stratify=Y,
                                                    test_size=0.25, random_state=0) # random_state will keep the output of the "random" sample equal Delete the parameter to make the sample actually random.



##############################################
# Part 4: Running and configuring the models #
##############################################

report = pd.DataFrame(columns=['Model','Acc.Train','Acc.Test'])

# k-Nearest Neighbors (kNN)
n_neighbors_config = 6
n_neighbors_end_of_range_config = 15

# Decision tree
decision_tree_end_of_range_config = 20
decision_tree_max_depth_config = 5

# Random forest
random_forest_end_of_range_max_depth_config = 5
random_forest_end_of_range_n_estimators_config = 5
random_forest_max_depth_config = 7

# Gradient boosting
gradient_boosting_max_depth_maximum_config = 10
gradient_boosting_learning_rate_maximum_config = 10

# Neural network
neural_network_hidden_layer_size_maximum_config = 10

# Cross validation
cross_validation_number_of_folds = 4


####################
# Cross Validation #
####################

from sklearn.metrics import accuracy_score

print(df['target90'].value_counts())
df_majority = df[df.target90==0]
df_minority = df[df.target90==1]
from sklearn.utils import resample
df_majority_balanced = resample(df_majority,
                                n_samples=len(df_minority),
                                replace=False)
df_balanced = pd.concat([df_majority_balanced,
                                  df_minority])
print("\n \n Cross Validation")
print(df_balanced['target90'].value_counts())

#Partitioning
X = df_balanced.drop('target90', axis = 1)
Y = df_balanced['target90']
print(Y.value_counts())

## 1 random sampling siplitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y,
                                                    test_size=0.20, random_state=0)
print(Y_train.value_counts())
print(Y_test.value_counts())

report_x = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation','Acc. Test'])


#######################
# Logistic Regression #
#######################

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(solver = 'lbfgs')
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(lrmodel, X_train, Y_train, scoring='accuracy', cv=cross_validation_number_of_folds)
print("\n \n  \n Logistic Regression  \n")
print("Accuracies = ", accuracies)
print("Mean = ", accuracies.mean())
print("SD = ", accuracies.std())
lrmodel.fit(X_train, Y_train)
Y_test_pred = lrmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report_x.loc[len(report_x)] = ['Logistic Regression', accuracies.mean(), accuracies.std(), accte]
print(report_x.loc[len(report_x)-1])


###############
# Naive Bayes #
###############

from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(nbmodel, X_train, Y_train, scoring='accuracy', cv=cross_validation_number_of_folds)
print("\n \n  \nNaive Bayes  \n")
print("Accuracies = ", accuracies)
nbmodel.fit(X_train, Y_train)
Y_test_pred = nbmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report_x.loc[len(report_x)] = ['Naive Bayes', accuracies.mean(), accuracies.std(), accte]
print(report_x.loc[len(report_x)-1])


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [4., 5., 6., 7., 8., 9., 10., 11., 12.],
    'n_estimators': [100]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, cv=cross_validation_number_of_folds)
CV_rfmodel.fit(X_train, Y_train)
print("\n \n \n Random Forest \n")
print("Best parameters: ", CV_rfmodel.best_params_)
# Apply the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
Y_test_pred = rfmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report_x.loc[len(report_x)] = ['Random Forest',
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_],
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], accte]
print(report_x.loc[len(report_x)-1])

plt.plot(range(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1] + 1)), CV_rfmodel.cv_results_['mean_test_score'])
plt.xlim(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1])-1)
plt.xticks(range(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1])))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Random Forest: Comparison of Accuracies (n_estimators const. at 100)')
plt.show()


################################
# Gradient Boosting Classifier #
################################

from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [2., 3., 4.],
    'subsample': [0.8],
    'n_estimators': [100],
    'learning_rate': [0.2]
}
CV_gbmodel = GridSearchCV(estimator=gbmodel, param_grid=param_grid, cv=cross_validation_number_of_folds)
CV_gbmodel.fit(X_train, Y_train)
print("\n \n \n Gradient Boosting Classifier  \n")
print(CV_gbmodel.best_params_)
#use the best parameters
gbmodel = gbmodel.set_params(**CV_gbmodel.best_params_)
gbmodel.fit(X_train, Y_train)
Y_test_pred = gbmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report_x.loc[len(report_x)] = ['Gradient Boosting (grid)',
                          CV_gbmodel.cv_results_['mean_test_score'][CV_gbmodel.best_index_],
                          CV_gbmodel.cv_results_['std_test_score'][CV_gbmodel.best_index_], accte]
print(report_x.loc[len(report_x)-1])

print(CV_rfmodel.cv_results_['mean_test_score'])

plt.plot(range(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1] + 1)), CV_gbmodel.cv_results_['mean_test_score'])
plt.xlim(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1])-1)
plt.xticks(range(int(param_grid['max_depth'][0]), int(param_grid['max_depth'][-1])))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting: Comparison of Accuracies')
plt.show()


################
# Final Report #
################

# Possible biases

# Potential for improvement

# Is the method feasible in practice?
# Yes, the media dealer aggregated over 30,000 rows of data about their customers. We therefore conclude that the marketing department could implement a data analytics model and use it on a regular basis for this specific purpose. The computing power needed is limited and the person that performs the analysis could work with a script that runs all steps from preprocessing to reporting. A qualified data analyst may be necessary to deal with changes in the data input and to refine the model (e.g., if new interesting variables are added to the ERP system). Scalability should not be a problem. In addition to implementing a predictive system, there should also be an evalution process in place that makes sure that the results are measured and compared to actual outcomes. Also, one should measure if doing the prediction actually pays of enough (cost for making the predictions vs. gains from better decisions).
# It would be interesting to know what the "Model" variable measures. We dropped this variable because we could not check the value for plausibility.

print(report)
