import joblib as jb
import numpy as np
import json
import os

# loading BayesSearch object already fited
bs = jb.load("model/bs_fitted.pkl")

# Loading X_train and y_train together but then saving only y_train
X_train, y_train = jb.load("paper_code_and_files/saved_files/training_data.pkl")
jb.dump(y_train, "model/y_train_only.pkl")

# Separating only the classifier and saving only it's parameters
clf = bs.best_estimator_["LR"]
jb.dump(clf.coef_, "model/clf_coef.pkl")
jb.dump(clf.intercept_, "model/clf_intercept.pkl")

# function to save fragmented pca
def save_fragmented_pca(pca, directory='fragmented_pca', components_parts=5):
    """
    Saves the PCA attributes into fragmented files.
    
    Parameters:
    - pca: Already fitted PCA object.
    - directory: Directory where the fragments will be saved.
    - components_parts: Number of parts to subdivide `components_`.
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save attributes that are not `components_`
    np.save(os.path.join(directory, 'explained_variance_.npy'), pca.explained_variance_)
    np.save(os.path.join(directory, 'explained_variance_ratio_.npy'), pca.explained_variance_ratio_)
    np.save(os.path.join(directory, 'singular_values_.npy'), pca.singular_values_)
    np.save(os.path.join(directory, 'mean_.npy'), pca.mean_)
    
    # Subdivide `components_` into parts
    components = pca.components_
    n_parts = components_parts
    subdivided_components = np.array_split(components, n_parts, axis=1)  # Split by columns
    
    for i, part in enumerate(subdivided_components):
        np.save(os.path.join(directory, f'components_part_{i+1}.npy'), part)
    
    # Save additional parameters in a JSON file
    pca_params = {
        'n_components': pca.n_components,
        'svd_solver': pca.svd_solver,
        'tol': pca.tol,
        'copy': pca.copy,
        'whiten': pca.whiten,
        'random_state': pca.random_state,
        'components_parts': components_parts
        # Add other parameters as needed
    }
    
    with open(os.path.join(directory, 'pca_params.json'), 'w') as f:
        json.dump(pca_params, f)

# Separating only the PCA and saving it
pca = bs.best_estimator_["PCA"]
save_fragmented_pca(pca, directory='model/fragmented_pca', components_parts=5)

# Getting train_log_odds and saving it (its lighter)
train_log_odds = clf.intercept_ + np.sum(pca.transform(X_train) * clf.coef_, axis = 1)
jb.dump(train_log_odds, "model/train_log_odds.pkl")