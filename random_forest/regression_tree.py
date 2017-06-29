import numpy as np
from .tree import TreeParams, build_tree, _predict, _feature_importances


class RegressionTree:
    """A decision tree with continuous predictors and response variables."""
    
    def __init__(self, X, y,
                 max_depth=None,
                 max_features=None,
                 min_samples_split=None,
                 min_samples_leaf=None,
                 random_seed=None):
        """Fit a regression tree to continuous data.

        Arguments
        ---------
        X : (n, m) array
            The predictor variables.
        y : (n, ) array
            The response variable to model.
        max_depth : [inf] | int | float
            The maximum depth of the tree.
        min_samples_split : [0] | int
            The minimum number of measurements required to split a node.
        min_samples_leaf : int | float
            The minimum number of samples required in each leaf node.
        random_seed : int
            For reproducibility.

        Notes
        -----
        This regression tree currently only supports depth-first construction.
        As such, the following scikit-learn DecisionTreeRegressor arguments
        are not supported:
            `min_weight_fraction_leaf`, `max_leaf_nodes`

        The arguments `min_impurity_split` and `presort` are also not supported.
        """
        
        # Ensure `X` and `y` are the right shape
        n, n_features = X.shape
        self._n, self._n_features = n, n_features
        
        if y.ndim is not 1:
            raise ValueError("`y` must be a 1-dimensional array.")
        elif y.size != n:
            raise ValueError("`y` must have the same size as `X` over dim 1.")
        
        # Validate the input parameters
        params = TreeParams()
        
        # max_depth
        if max_depth is None:
            max_depth = n
        elif type(max_depth) is not int:
            raise TypeError("`max_depth` should be an int or None.")
        # print('max_depth', max_depth)
        params.max_depth = max_depth
        
        # min_samples_leaf
        if min_samples_leaf is None:
            min_samples_leaf = 1
        elif type(min_samples_leaf) is not int:
            raise TypeError("`min_samples_leaf` must be an int or None.")
        elif min_samples_leaf < 0:
            raise ValueError("`min_samples_leaf` must be >= 1.")
        params.min_samples_leaf = min_samples_leaf
        
        # min_samples_split
        if type(min_samples_split) is int:
            if min_samples_split < 2:
                raise ValueError()
            min_samples_split
        elif min_samples_split is None:
            min_samples_split = 2
        params.min_samples_split = min_samples_split
        
        # max_features
        if max_features is None:
            max_features = 0.3333333
        
        if type(max_features) is float:
            max_features = max(1, int(np.round(n_features * max_features)))
        elif type(max_features) is int:
            if max_features <= 0:
                raise ValueError("`max_features` must be >= 1.")
            elif max_features > n_features:
                raise ValueError(
                    "`max_features` must be <= n_features (X.shape[1]).")
        params.max_features = max_features
        
        # print('md', params.max_depth)
        # print('msl', params.min_samples_leaf)
        # print('mss', params.min_samples_split)
        # print('max_features', max_features)
        # print('n, n_features', X.shape, y.shape)
        
        # random_seed
        if random_seed is None:
            random_seed = np.random.randint(np.iinfo(np.int32).max)
        # print('random_seed', random_seed)
        
        how = np.zeros(n_features, dtype=np.uint8)
        
        #self._n, self._n_features = X.shape
        self._params = params
        self._n_nodes, self._flat_tree = build_tree(
            X.copy(), y.copy(), params, how, random_seed)
        self._flat_tree.flags.writeable = False

    @property
    def n(self):
        return self._n
    
    @property
    def n_nodes(self):
        return self._n_nodes
    
    @property
    def n_features(self):
        return self._n_features

    @property
    def max_depth(self):
        return self._params.max_depth
    
    @property
    def max_features(self):
        return self._params.max_features

    @property
    def min_samples_split(self):
        return self._params.min_samples_split

    @property
    def min_samples_leaf(self):
        return self._params.min_samples_leaf
    
    @property
    def flat_tree(self):
        return self._flat_tree
    
    def predict(self, X):
        """Predict the values of y at a given X values."""
        return _predict(self._flat_tree, X)
    
    def feature_importances(self):
        """Calculate proxy variable importance fo"""
        return _feature_importances(self._flat_tree, self.n_features)