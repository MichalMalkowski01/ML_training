import numpy as np
from sklearn.naive_bayes import BernoulliNB
def get_label_indices(labels):
    """
    Group the indices based on their labels and return index
    @param labels: list of labels
    @return: dict of indices, {class1: [index], class2: [index]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    return label_indices

def get_prior(label_indices):
    """
    Get the prior probability based on training data
    @param label_indices: dict of indices, {class1: [index], class2: [index]}
    @return: dict of prior probabilities, {class1: prior, class2: prior}
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

def get_likelihood(features, label_indices, smoothing=0):
    """
    Get the likelihood based on training data
    @param features: 2D array of features, shape (n_samples, n_features)
    @param label_indices: dict of indices, {class1: [index], class2: [index]}
    @param smoothing: smoothing parameter for Laplace smoothing
    @return: dict of likelihoods, {class1: [likelihoods], class2: [likelihoods]}
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)  # Assuming binary features
    return likelihood

def get_posterior(X, prior, likelihood):
    """
    Get the posterior probability based on the prior and likelihood
    @param X: training data, shape (n_samples, n_features)
    @param prior: dict of prior probabilities, {class1: prior, class2: prior}
    @param likelihood: dict of likelihoods, {class1: [likelihoods], class2: [likelihoods]}
    @return: dict of posterior probabilities, {class1: [posteriors], class2: [posteriors]}
    """
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


def main():
    X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
    ])
    Y_train = ['T', 'N', 'T', 'T']
    X_test = np.array([[1, 1, 0]])
    label_indices = get_label_indices(Y_train)
    print("label_indices: ", label_indices)
    prior = get_prior(label_indices)
    print("prior: ", prior)
    smoothing = 1
    likelihood = get_likelihood(X_train, label_indices, smoothing)
    print("likelihood: ", likelihood)
    posterior = get_posterior(X_test, prior, likelihood)
    print("posterior: ", posterior)
    
    clf = BernoulliNB(alpha=1, fit_prior=True)
    clf.fit(X_train, Y_train)
    pred_prob = clf.predict_proba(X_test)
    print("[sktlearn] pred_prob: ", pred_prob)
    pred = clf.predict(X_test)
    print("[sktlearn] pred: ", pred)
    

    
    
if __name__ == "__main__":    main()