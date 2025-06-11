import numpy as np
from scipy import sparse

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        # Clip z để tránh overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _get_batches(self, X, y, batch_size):
        """Generator để chia dữ liệu thành batches"""
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            if sparse.issparse(X):
                yield X[i:end_idx], y[i:end_idx]
            else:
                yield X[i:end_idx], y[i:end_idx]
    
    def fit(self, X, y):
        # Convert labels to 0/1 if they're strings
        if isinstance(y.iloc[0], str):
            y = (y == 'spam').astype(int)
        
        n_samples, n_features = X.shape
        
        # Initialize weights và bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert to numpy array if needed
        if hasattr(y, 'values'):
            y = y.values
            
        prev_cost = float('inf')
        
        for iteration in range(self.max_iter):
            total_cost = 0
            n_batches = 0
            
            # Shuffle indices for each epoch
            indices = np.random.permutation(n_samples)
            if sparse.issparse(X):
                X_shuffled = X[indices]
            else:
                X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process each batch
            for X_batch, y_batch in self._get_batches(X_shuffled, y_shuffled, self.batch_size):
                batch_size = X_batch.shape[0]
                
                # Forward pass
                if sparse.issparse(X_batch):
                    z = X_batch.dot(self.weights) + self.bias
                else:
                    z = np.dot(X_batch, self.weights) + self.bias
                
                predictions = self._sigmoid(z)
                
                # Compute cost for this batch
                epsilon = 1e-15  # Để tránh log(0)
                predictions = np.clip(predictions, epsilon, 1 - epsilon)
                batch_cost = -np.mean(y_batch * np.log(predictions) + 
                                    (1 - y_batch) * np.log(1 - predictions))
                total_cost += batch_cost * batch_size
                n_batches += batch_size
                
                # Compute gradients
                dz = predictions - y_batch
                if sparse.issparse(X_batch):
                    dw = X_batch.T.dot(dz) / batch_size
                else:
                    dw = np.dot(X_batch.T, dz) / batch_size
                db = np.mean(dz)
                
                # Update weights và bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Average cost across all batches
            avg_cost = total_cost / n_batches
            
            # Check for convergence
            if abs(prev_cost - avg_cost) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
                
            prev_cost = avg_cost
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Cost: {avg_cost:.6f}")
    
    def predict_proba(self, X):
        """Predict probabilities with batch processing"""
        n_samples = X.shape[0]
        probabilities = np.zeros(n_samples)
        
        start_idx = 0
        for X_batch, _ in self._get_batches(X, np.zeros(n_samples), self.batch_size):
            batch_size = X_batch.shape[0]
            
            if sparse.issparse(X_batch):
                z = X_batch.dot(self.weights) + self.bias
            else:
                z = np.dot(X_batch, self.weights) + self.bias
            
            batch_proba = self._sigmoid(z)
            probabilities[start_idx:start_idx + batch_size] = batch_proba
            start_idx += batch_size
            
        return probabilities
    
    def predict(self, X):
        """Predict with batch processing"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)