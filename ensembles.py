import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.args = trees_parameters
        self.estimators = [
            DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)
            for i in range(n_estimators)
        ]
        self.estimator_features = [0] * n_estimators
        self.n_algorithms = 0

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        N = X.shape[0]  # Вообще говоря, размер выборки - l
        d = X.shape[1]
        if self.feature_subsample_size is None:
            subspace_size = d // 3
        else:
            subspace_size = int(d * self.feature_subsample_size)

        if X_val is not None:
            MSE_history = np.zeros(self.n_estimators)

        for i in range(self.n_estimators):

            # sample bagging
            samples = np.random.choice(N, size=N, replace=True)

            # selecting random subspace
            features = np.random.choice(
                X.shape[1], size=subspace_size, replace=False
                )
            self.estimator_features[i] = features
            X_train = X[:, features][samples]

            # fitting i-th algorithm
            self.estimators[i].fit(X_train, y[samples])

            self.n_algorithms = i + 1
            if X_val is not None:
                MSE_history[i] = np.mean(
                    (y_val - self.predict(X_val))**2
                    )

        if X_val is not None:
            return MSE_history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        N = X.shape[0]
        y = np.zeros(N)
        # taking mean of algorithm predictions
        for i in range(self.n_algorithms):
            y = y + self.estimators[i].predict(X[:,
                                                 self.estimator_features[i]])
        return y / self.n_algorithms


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5,
        feature_subsample_size=None, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_subsample_size = feature_subsample_size
        self.args = trees_parameters
        self.estimators = [
            DecisionTreeRegressor(**trees_parameters)
            for i in range(n_estimators)
        ]
        self.estimator_features = [0] * n_estimators
        self.alpha_array = np.zeros(self.n_estimators)
        self.n_algorithms = 0

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        N = X.shape[0]
        d = X.shape[1]

        if X_val is not None:
            MSE_history = np.zeros(self.n_estimators)

        # we shall save f-array for improved efficiency
        f = np.zeros(N)

        if self.feature_subsample_size is None:
            subspace_size = d // 3
        else:
            subspace_size = int(d * self.feature_subsample_size)

        for i in range(self.n_estimators):

            # selecting random subspace
            features = np.random.choice(
                X.shape[1], size=subspace_size, replace=False
            )
            self.estimator_features[i] = features
            X_train = X[:, features]

            # First step
            # Q = MSE(f, y) => gradQ_f = f_{T-1, i} - y_i
            self.estimators[i].fit(X_train, y - f)
            b = self.estimators[i].predict(X_train)

            # Second step
            res = minimize_scalar(
                lambda x: np.sum((f + x * b - y)**2),
                bounds=(0, 1e4), method='bounded'
                )
            self.alpha_array[i] = self.learning_rate * res.x

            # Final step
            f = f + self.alpha_array[i] * b

            self.n_algorithms = i + 1
            if X_val is not None:
                MSE_history[i] = np.mean(
                    (y_val - self.predict(X_val))**2
                    )

        if X_val is not None:
            return MSE_history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        N = X.shape[0]

        y = np.zeros(N)
        for i in range(self.n_algorithms):
            y = y + (
                self.estimators[i].predict(X[:, self.estimator_features[i]]) *
                self.alpha_array[i]
            )
        return y
