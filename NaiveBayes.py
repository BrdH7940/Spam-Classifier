import pandas as pd
import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None # (Spam / Ham)
        self.class_prior = None # (Prob(Spam) / Prob(Ham))
        self.feature_probs = None # (Prob(word | Spam) / Prob(word | Ham))

    def fit(self, X, y):
        '''
        Input:
            X: (n_samples, n_features)
            y: (n_samples,)
        '''
        n_samples, n_features = X.shape

        # Calculate class priors P(y)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize class_prior array
        self.class_prior = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            self.class_prior[i] = np.sum(y == c) / n_samples

        # Calculate feature probabilities P(x|y) with Laplace smoothing
        self.feature_probs = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            self.feature_probs[i, :] = (np.sum(X[y == c], axis=0) + self.alpha) / (np.sum(y == c) + n_features * self.alpha)

    def predict(self, X):
        '''
        Input:
            X: (n_samples, n_features)
        '''
        n_samples, n_features = X.shape

        # Calculate class posteriors P(y|x)
        posteriors = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            posteriors[:, i] = np.log(self.class_prior[i])
            # For multinomial NB, we only consider presence (X > 0) of features
            posteriors[:, i] += np.sum(np.log(self.feature_probs[i]) * X, axis=1)

        return self.classes[np.argmax(posteriors, axis=1)]