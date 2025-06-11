import numpy as np
from scipy import sparse
from collections import defaultdict
import random

class TreeNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False
        
class XGBoostTree:
    def __init__(self, max_depth=3, min_child_weight=0.1, reg_lambda=1.0, 
                 reg_alpha=0.0, gamma=0.0, colsample_bytree=1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.root = None
        self.feature_indices = None
        
    def _calculate_leaf_weight(self, gradients, hessians):
        """Calculate optimal leaf weight using Newton-Raphson"""
        grad_sum = np.sum(gradients)
        hess_sum = np.sum(hessians)
        
        if hess_sum == 0:
            return 0
            
        weight = -grad_sum / (hess_sum + self.reg_lambda)
        
        # Apply L1 regularization (soft thresholding)
        if self.reg_alpha > 0:
            if weight > self.reg_alpha:
                weight -= self.reg_alpha
            elif weight < -self.reg_alpha:
                weight += self.reg_alpha
            else:
                weight = 0
                
        return weight
    
    def _calculate_gain(self, left_grad, left_hess, right_grad, right_hess, parent_grad, parent_hess):
        """Calculate gain from split"""
        def calculate_score(grad_sum, hess_sum):
            if hess_sum == 0:
                return 0
            return (grad_sum ** 2) / (hess_sum + self.reg_lambda)
        
        left_score = calculate_score(left_grad, left_hess)
        right_score = calculate_score(right_grad, right_hess)
        parent_score = calculate_score(parent_grad, parent_hess)
        
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        return gain
    
    def _get_feature_column_sparse(self, X, feature_idx):
        """Efficiently get feature column from sparse matrix"""
        if sparse.issparse(X):
            return X[:, feature_idx].toarray().flatten()
        else:
            return X[:, feature_idx]
    
    def _find_best_split(self, X, gradients, hessians):
        """Find best split for current node - memory efficient version"""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        n_samples, n_features = X.shape
        
        # Sample features for this tree (column sampling) - more aggressive
        if self.feature_indices is None:
            n_features_sample = max(1, int(n_features * self.colsample_bytree))
            # Allow more features but still limit for memory
            n_features_sample = min(n_features_sample, 500)  # Increased from 100
            self.feature_indices = np.random.choice(n_features, n_features_sample, replace=False)
        
        # Try each sampled feature
        for feature_idx in self.feature_indices:
            # Get feature values efficiently
            feature_values = self._get_feature_column_sparse(X, feature_idx)
            
            # Skip if all values are the same
            if len(np.unique(feature_values)) <= 1:
                continue
            
            # More thresholds for better splits
            unique_values = np.unique(feature_values)
            if len(unique_values) > 20:
                # Use more quantiles for better splits
                percentiles = np.linspace(5, 95, 19)  # More percentiles
                thresholds = np.percentile(unique_values, percentiles)
                thresholds = np.unique(thresholds)
            elif len(unique_values) > 10:
                # Use all unique values for medium-sized sets
                thresholds = unique_values[:-1]
            else:
                thresholds = unique_values[:-1]
            
            for threshold in thresholds:
                # Split samples
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum child weight constraint
                left_hess_sum = np.sum(hessians[left_mask])
                right_hess_sum = np.sum(hessians[right_mask])
                
                if left_hess_sum < self.min_child_weight or right_hess_sum < self.min_child_weight:
                    continue
                
                # Calculate gain
                left_grad_sum = np.sum(gradients[left_mask])
                right_grad_sum = np.sum(gradients[right_mask])
                parent_grad_sum = np.sum(gradients)
                parent_hess_sum = np.sum(hessians)
                
                gain = self._calculate_gain(left_grad_sum, left_hess_sum,
                                          right_grad_sum, right_hess_sum,
                                          parent_grad_sum, parent_hess_sum)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = np.where(left_mask)[0]
                    best_right_indices = np.where(right_mask)[0]
        
        return best_gain, best_feature, best_threshold, best_left_indices, best_right_indices
    
    def _build_tree(self, X, gradients, hessians, depth=0, indices=None):
        """Recursively build tree - memory efficient version"""
        if indices is None:
            indices = np.arange(len(gradients))
        
        node = TreeNode()
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(indices) <= 1 or 
            np.sum(hessians[indices]) < self.min_child_weight):
            
            # Create leaf node
            node.is_leaf = True
            node.value = self._calculate_leaf_weight(gradients[indices], hessians[indices])
            return node
        
        # Get subset of data for current node
        if sparse.issparse(X):
            X_subset = X[indices]
        else:
            X_subset = X[indices]
        grad_subset = gradients[indices]
        hess_subset = hessians[indices]
        
        # Find best split
        gain, feature_idx, threshold, left_rel_indices, right_rel_indices = \
            self._find_best_split(X_subset, grad_subset, hess_subset)
        
        # If no good split found, create leaf
        if gain <= 0 or feature_idx is None:
            node.is_leaf = True
            node.value = self._calculate_leaf_weight(grad_subset, hess_subset)
            return node
        
        # Create internal node
        node.feature_idx = feature_idx
        node.threshold = threshold
        
        # Convert relative indices back to absolute indices
        left_indices = indices[left_rel_indices]
        right_indices = indices[right_rel_indices]
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X, gradients, hessians, depth + 1, left_indices)
        node.right = self._build_tree(X, gradients, hessians, depth + 1, right_indices)
        
        return node
    
    def fit(self, X, gradients, hessians):
        """Fit tree to gradients and hessians"""
        self.root = self._build_tree(X, gradients, hessians)
    
    def _predict_single_sparse(self, x_sparse, node):
        """Predict single sparse sample recursively"""
        if node.is_leaf:
            return node.value
        
        # Get feature value from sparse vector
        if sparse.issparse(x_sparse):
            feature_value = x_sparse[0, node.feature_idx]
        else:
            feature_value = x_sparse[node.feature_idx]
        
        if feature_value <= node.threshold:
            return self._predict_single_sparse(x_sparse, node.left)
        else:
            return self._predict_single_sparse(x_sparse, node.right)
    
    def predict(self, X):
        """Predict for multiple samples - memory efficient"""
        predictions = np.zeros(X.shape[0])
        
        if sparse.issparse(X):
            # Process sparse matrix row by row to save memory
            for i in range(X.shape[0]):
                x_row = X[i:i+1]  # Keep as sparse
                predictions[i] = self._predict_single_sparse(x_row, self.root)
        else:
            for i, x in enumerate(X):
                predictions[i] = self._predict_single_sparse(x, self.root)
        
        return predictions

class XGBoost:
    def __init__(self, n_estimators=50, learning_rate=0.3, max_depth=6, 
                 min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=0.1, reg_alpha=0.0, gamma=0.0, 
                 early_stopping_rounds=None, eval_metric='logloss',
                 batch_size=500, random_state=42):
        
        # Reduced defaults for memory efficiency
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.trees = []
        self.base_prediction = 0
        self.train_scores = []
        self.val_scores = []
        self.best_iteration = 0
        
        # Set random seed
        np.random.seed(random_state)
        random.seed(random_state)
        
    def _sigmoid(self, z):
        """Sigmoid function with overflow protection"""
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _compute_gradients_hessians(self, y_true, y_pred):
        """Compute gradients and hessians for logistic loss"""
        p = self._sigmoid(y_pred)
        gradients = p - y_true
        hessians = p * (1 - p)
        hessians = np.maximum(hessians, 1e-16)
        return gradients, hessians
    
    def _compute_loss(self, y_true, y_pred):
        """Compute logistic loss"""
        p = self._sigmoid(y_pred)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    
    def _get_batches(self, X, y=None, batch_size=None):
        """Generator để chia dữ liệu thành batches"""
        if batch_size is None:
            batch_size = self.batch_size
            
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            if y is not None:
                yield X[i:end_idx], y[i:end_idx]
            else:
                yield X[i:end_idx]
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Fit XGBoost model - memory efficient version"""
        # Convert labels to 0/1 if they're strings
        if hasattr(y_train, 'iloc') and isinstance(y_train.iloc[0], str):
            y_train = (y_train == 'spam').astype(int)
        if hasattr(y_train, 'values'):
            y_train = y_train.values
            
        if X_val is not None and y_val is not None:
            if hasattr(y_val, 'iloc') and isinstance(y_val.iloc[0], str):
                y_val = (y_val == 'spam').astype(int)
            if hasattr(y_val, 'values'):
                y_val = y_val.values
        
        n_samples = X_train.shape[0]
        
        # Initialize base prediction (log odds)
        pos_rate = np.mean(y_train)
        pos_rate = np.clip(pos_rate, 1e-15, 1 - 1e-15)
        self.base_prediction = np.log(pos_rate / (1 - pos_rate))
        
        # Initialize predictions
        train_predictions = np.full(n_samples, self.base_prediction)
        if X_val is not None:
            val_predictions = np.full(X_val.shape[0], self.base_prediction)
        
        if verbose:
            print(f"Starting XGBoost training with {self.n_estimators} estimators...")
            print(f"Data shape: {X_train.shape}, Sparse: {sparse.issparse(X_train)}")
            print(f"Base prediction: {self.base_prediction:.4f}")
        
        best_val_score = float('inf')
        no_improvement_count = 0
        
        for estimator_idx in range(self.n_estimators):
            if verbose and (estimator_idx + 1) % 5 == 0:
                print(f"Training estimator {estimator_idx + 1}/{self.n_estimators}")
            
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradients_hessians(y_train, train_predictions)
            
            # Row sampling (subsample) - less aggressive for better learning
            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                # Increase subsample size for better learning
                n_subsample = min(n_subsample, 8000)  # Increased from 5000
                sample_indices = np.random.choice(n_samples, n_subsample, replace=False)
                
                X_subsample = X_train[sample_indices]
                gradients_subsample = gradients[sample_indices]
                hessians_subsample = hessians[sample_indices]
            else:
                # Allow larger sample size for better learning
                if n_samples > 8000:  # Increased from 5000
                    sample_indices = np.random.choice(n_samples, 8000, replace=False)
                    X_subsample = X_train[sample_indices]
                    gradients_subsample = gradients[sample_indices]
                    hessians_subsample = hessians[sample_indices]
                else:
                    X_subsample = X_train
                    gradients_subsample = gradients
                    hessians_subsample = hessians
            
            # Create and fit tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                colsample_bytree=self.colsample_bytree
            )
            
            tree.fit(X_subsample, gradients_subsample, hessians_subsample)
            self.trees.append(tree)
            
            # Update predictions using batch processing
            # Training predictions
            start_idx = 0
            for X_batch in self._get_batches(X_train, batch_size=self.batch_size):
                batch_size = X_batch.shape[0]
                tree_pred = tree.predict(X_batch)
                train_predictions[start_idx:start_idx + batch_size] += self.learning_rate * tree_pred
                start_idx += batch_size
            
            # Validation predictions
            if X_val is not None:
                start_idx = 0
                for X_batch in self._get_batches(X_val, batch_size=self.batch_size):
                    batch_size = X_batch.shape[0]
                    tree_pred = tree.predict(X_batch)
                    val_predictions[start_idx:start_idx + batch_size] += self.learning_rate * tree_pred
                    start_idx += batch_size
            
            # Compute and store scores
            train_loss = self._compute_loss(y_train, train_predictions)
            self.train_scores.append(train_loss)
            
            if X_val is not None:
                val_loss = self._compute_loss(y_val, val_predictions)
                self.val_scores.append(val_loss)
                
                # Early stopping
                if self.early_stopping_rounds is not None:
                    if val_loss < best_val_score:
                        best_val_score = val_loss
                        self.best_iteration = estimator_idx
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= self.early_stopping_rounds:
                        if verbose:
                            print(f"Early stopping at iteration {estimator_idx + 1}")
                            print(f"Best iteration: {self.best_iteration + 1}")
                        break
                
                if verbose and (estimator_idx + 1) % 5 == 0:
                    print(f"Iteration {estimator_idx + 1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (estimator_idx + 1) % 5 == 0:
                    print(f"Iteration {estimator_idx + 1}: Train Loss: {train_loss:.6f}")
        
        if verbose:
            print("Training completed!")
    
    def predict_proba(self, X):
        """Predict probabilities with batch processing"""
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.base_prediction)
        
        # Determine how many trees to use (for early stopping)
        n_trees_to_use = len(self.trees)
        if (self.early_stopping_rounds is not None and 
            hasattr(self, 'best_iteration') and 
            self.best_iteration < len(self.trees)):
            n_trees_to_use = self.best_iteration + 1
        
        # Add predictions from each tree using batch processing
        for i, tree in enumerate(self.trees[:n_trees_to_use]):
            start_idx = 0
            for X_batch in self._get_batches(X, batch_size=self.batch_size):
                batch_size = X_batch.shape[0]
                tree_pred = tree.predict(X_batch)
                predictions[start_idx:start_idx + batch_size] += self.learning_rate * tree_pred
                start_idx += batch_size
        
        # Convert to probabilities
        probabilities = self._sigmoid(predictions)
        return probabilities
    
    def predict(self, X):
        """Predict with batch processing"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def get_feature_importance(self, importance_type='gain'):
        """Get feature importance (simplified version)"""
        feature_importance = defaultdict(float)
        
        for tree in self.trees:
            self._traverse_tree_for_importance(tree.root, feature_importance)
        
        return dict(feature_importance)
    
    def _traverse_tree_for_importance(self, node, importance_dict):
        """Traverse tree to calculate feature importance"""
        if node is None or node.is_leaf:
            return
        
        importance_dict[node.feature_idx] += 1
        
        self._traverse_tree_for_importance(node.left, importance_dict)
        self._traverse_tree_for_importance(node.right, importance_dict)
