class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []  
        
    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _get_batches(self, X, y, batch_size):
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            if sparse.issparse(X):
                yield X[i:end_idx], y[i:end_idx]
            else:
                yield X[i:end_idx], y[i:end_idx]
    
    def fit(self, X, y, X_val=None, y_val=None):
        if isinstance(y.iloc[0], str):
            y = (y == 'spam').astype(int)
        
        if X_val is not None and y_val is not None:
            if isinstance(y_val.iloc[0], str):
                y_val = (y_val == 'spam').astype(int)
            if hasattr(y_val, 'values'):
                y_val = y_val.values
        
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if hasattr(y, 'values'):
            y = y.values
            
        prev_cost = float('inf')
        self.loss_history = []  
        self.val_loss_history = [] 
        
        for iteration in range(self.max_iter):
            total_cost = 0
            n_batches = 0
            
            indices = np.random.permutation(n_samples)
            if sparse.issparse(X):
                X_shuffled = X[indices]
            else:
                X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for X_batch, y_batch in self._get_batches(X_shuffled, y_shuffled, self.batch_size):
                batch_size = X_batch.shape[0]
                
                if sparse.issparse(X_batch):
                    z = X_batch.dot(self.weights) + self.bias
                else:
                    z = np.dot(X_batch, self.weights) + self.bias
                
                predictions = self._sigmoid(z)
                
                epsilon = 1e-15
                predictions = np.clip(predictions, epsilon, 1 - epsilon)
                batch_cost = -np.mean(y_batch * np.log(predictions) + 
                                    (1 - y_batch) * np.log(1 - predictions))
                total_cost += batch_cost * batch_size
                n_batches += batch_size
                
                dz = predictions - y_batch
                if sparse.issparse(X_batch):
                    dw = X_batch.T.dot(dz) / batch_size
                else:
                    dw = np.dot(X_batch.T, dz) / batch_size
                db = np.mean(dz)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            avg_cost = total_cost / n_batches
            self.loss_history.append(avg_cost)
            
            if X_val is not None and y_val is not None:
                val_loss = self._calculate_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)
            
            if abs(prev_cost - avg_cost) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
                
            prev_cost = avg_cost
            
            if (iteration + 1) % 100 == 0:
                val_info = f", Val Loss: {self.val_loss_history[-1]:.6f}" if X_val is not None else ""
                print(f"Iteration {iteration + 1}, Train Loss: {avg_cost:.6f}{val_info}")
    
    def _calculate_loss(self, X, y):
        if sparse.issparse(X):
            z = X.dot(self.weights) + self.bias
        else:
            z = np.dot(X, self.weights) + self.bias
        
        predictions = self._sigmoid(z)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    def predict_proba(self, X):
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
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def plot_loss(self, figsize=(12, 5)):
        if not self.loss_history:
            print("No loss history available. Train the model first.")
            return
        
        if len(self.val_loss_history) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            ax1.plot(self.loss_history, label='Training Loss', color='blue', linewidth=2)
            ax1.plot(self.val_loss_history, label='Validation Loss', color='red', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training vs Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.semilogy(self.loss_history, label='Training Loss', color='blue', linewidth=2)
            ax2.semilogy(self.val_loss_history, label='Validation Loss', color='red', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss (log scale)')
            ax2.set_title('Loss Convergence (Log Scale)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            ax1.plot(self.loss_history, color='blue', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            
            ax2.semilogy(self.loss_history, color='blue', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss (log scale)')
            ax2.set_title('Training Loss (Log Scale)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Initial Loss: {self.loss_history[0]:.6f}")
        print(f"Final Loss: {self.loss_history[-1]:.6f}")
        print(f"Loss Reduction: {((self.loss_history[0] - self.loss_history[-1]) / self.loss_history[0] * 100):.2f}%")
        print(f"Total Iterations: {len(self.loss_history)}")

def predict_csv(csv_path, vectorizer, model, label_col='Spam/Ham'):
    """
    Predict and evaluate on a CSV file with columns: Subject, Message, Spam/Ham.
    Args:
        csv_path (str): Path to CSV file
        vectorizer (CountVectorizer): Fitted vectorizer
        model (LogisticRegression): Trained model
        label_col (str): Name of label column
    Returns:
        dict: Metrics (accuracy, precision, recall, f1_score, confusion_matrix)
    """
    df = pd.read_csv(csv_path)
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['text'] = df['Subject'] + ' ' + df['Message']
    X = vectorizer.transform(df['text'])
    y_true = df[label_col].values  # Chuyển thành numpy array
    y_pred = model.predict(X)
    # Convert numeric prediction to label if needed
    if y_true.dtype.type is np.str_ or y_true.dtype.type is np.object_:
        y_pred_label = ['spam' if p == 1 else 'ham' for p in y_pred]
    else:
        y_pred_label = y_pred
    # Simple metrics
    accuracy = (y_pred_label == y_true).mean()
    print(f"Accuracy: {accuracy:.4f}")
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred_label
    }