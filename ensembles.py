import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None, random_state=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.r_state=random_state
        self.n = n_estimators
        self.max_depth = max_depth
        self.fss = feature_subsample_size
        self.trees_params = trees_parameters

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
        if not self.r_state is None:
            np.random.seed(self.r_state)
        if self.fss is None:
            self.ffs = X.shape[1] // 3
        self.forest = []
        self.forest_feat = []
        for i in range(self.n):
            feat = np.random.permutation(X.shape[1])[:self.ffs]
            objs = np.random.randint(0, X.shape[0], X.shape[0])
            tree = DecisionTreeRegressor(criterion='squared_error', max_depth=self.max_depth, **self.trees_params)
            tree.fit(X[:, feat][objs], y[objs])
            self.forest.append(tree)
            self.forest_feat.append(feat)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[0])
        for i in range(self.n):
            pred += self.forest[i].predict(X[:, self.forest_feat[i]])
        pred = pred/self.n
        return pred


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None, random_state=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.r_state = random_state
        self.n = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.fss = feature_subsample_size
        self.trees_params = trees_parameters

    def alpha_step(self, x, f, y, pred):
        return ((y - (f + x * pred))**2).sum()

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        if not self.r_state is None:
            np.random.seed(self.r_state)
        if self.fss is None:
            self.fss = X.shape[1] // 3
        self.forest = []
        self.forest_feat = []
        self.alphas = []
        init_f = np.zeros(y.shape[0])
        for i in range(self.n):
            feat = np.random.permutation(X.shape[1])[:self.fss]
            antigrad = 2 * (y - init_f)
            tree = DecisionTreeRegressor(criterion='squared_error', max_depth=self.max_depth, **self.trees_params)
            tree.fit(X[:, feat], antigrad)
            self.forest.append(tree)
            self.forest_feat.append(feat)
            pred = tree.predict(X[:, feat])
            alpha = minimize_scalar(self.alpha_step, args=(init_f, y, pred)).x
            self.alphas.append(alpha * self.lr)
            init_f += self.lr * alpha * pred

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[0])
        for i in range(self.n):
            pred += self.alphas[i] * self.forest[i].predict(X[:, self.forest_feat[i]])
        return pred