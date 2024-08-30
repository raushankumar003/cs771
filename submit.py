import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression



# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao
# train_data = pd.read_csv()
# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    X_tilda = my_map(X_train)
    model = LogisticRegression()
    model.fit(X_tilda, y_train)
    w = model.coef_.flatten()
    b = model.intercept_[0]
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    D_values = 1 - 2 * X  
    X1 = np.cumprod(D_values[:, ::-1], axis=1)[:, ::-1]
    features = [X1]
    i_upper = np.triu_indices(32, k=1)  
    pairwise_products = X1[:, :, np.newaxis] * X1[:, np.newaxis, :]
    pairwise_features = pairwise_products[:, i_upper[0], i_upper[1]]
    feat = np.hstack([X1, pairwise_features])
	
    return feat
