import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_=0.01, tolerance=1e-6, iters=10000, regularizer='l2'):
        """
        Initialize the logistic regression model.
        
        Parameters:
        learning_rate -- learning rate for gradient descent (default=0.01)
        lambda_ -- regularization strength (default=0.01)
        tolerance -- tolerance for early stopping (default=1e-6)
        iters -- maximum number of iterations (default=10000)
        regularizer -- type of regularization ('l1' or 'l2') (default='l2')
        """
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.iters = iters
        self.regularizer = regularizer  # 'l1' or 'l2'
        self.weights = None
        self.cost_history = []
    
    # Sigmoid function
    def sigmoid_activation(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    # Hypothesis function
    def develop_hypothesis(self, features):
        return self.sigmoid_activation(np.dot(features, self.weights))
    
    # Compute the cost with L1 or L2 regularization
    def compute_cost(self, hypothesis, target):
        num_examples = len(target)
        loss = -target * np.log(hypothesis + 1e-10) - (1 - target) * np.log(1 - hypothesis + 1e-10)
        
        if self.regularizer == 'l2':
            # L2 regularization term
            reg_term = (self.lambda_ / (2 * num_examples)) * np.sum(np.square(self.weights))
        elif self.regularizer == 'l1':
            # L1 regularization term
            reg_term = (self.lambda_ / num_examples) * np.sum(np.abs(self.weights))
        else:
            raise ValueError("Regularizer must be 'l1' or 'l2'")
        
        return np.mean(loss) + reg_term
    
    # Gradient Descent with L1 or L2 regularization
    def gradient_descent(self, features, target):
        num_examples = len(target)
        hypothesis = self.develop_hypothesis(features)
        gradient = np.dot(features.T, (hypothesis - target)) / num_examples
        
        if self.regularizer == 'l2':
            # L2 regularization: Add weight penalty
            gradient += (self.lambda_ / num_examples) * self.weights
        elif self.regularizer == 'l1':
            # L1 regularization: Add absolute weight penalty
            gradient += (self.lambda_ / num_examples) * np.sign(self.weights)
        
        return gradient
    
    # Update weights
    def update_weights(self, gradient):
        self.weights -= self.learning_rate * gradient
    
    # Training function with early stopping
    def fit(self, features, target):
        """
        Train the logistic regression model using gradient descent with L1 or L2 regularization.
        
        Parameters:
        features -- input features (numpy array)
        target -- target labels (numpy array)
        """
        num_features = features.shape[1]
        self.weights = np.zeros(num_features)  # Initialize weights to zeros
        prev_cost = float('inf')
        
        for i in range(self.iters):
            # Compute gradient and update weights
            gradient = self.gradient_descent(features, target)
            self.update_weights(gradient)
            
            # Compute cost
            hypothesis = self.develop_hypothesis(features)
            cost = self.compute_cost(hypothesis, target)
            self.cost_history.append(cost)
            
            # Early stopping based on cost improvement
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Early stopping at iteration {i} with cost {cost}")
                break
            prev_cost = cost

            # Print cost every 1000 iterations
            if i % 1000 == 0:
                print(f"Iteration {i}, Cost: {cost}")
        
        return self
    
    # Prediction function
    def predict(self, features, threshold=0.5):
        """
        Predict class labels for the input features.
        
        Parameters:
        features -- input features (numpy array)
        threshold -- decision threshold for classification (default=0.5)
        
        Returns:
        predictions -- predicted class labels (0 or 1)
        """
        probabilities = self.develop_hypothesis(features)
        return (probabilities >= threshold).astype(int)

    # Probability prediction
    def predict_proba(self, features):
        """
        Predict probabilities for the input features.
        
        Parameters:
        features -- input features (numpy array)
        
        Returns:
        probabilities -- predicted probabilities for each class
        """
        return self.develop_hypothesis(features)

