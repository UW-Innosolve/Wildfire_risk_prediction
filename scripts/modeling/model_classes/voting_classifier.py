# model_classes/voting_classifier.py

import numpy as np

class VotingClassifierCustom:
    def __init__(self, models, voting='soft'):
        """
        Initialize the VotingClassifierCustom with a dictionary of model instances.
        
        Args:
            models (dict): Dictionary with keys as model names and values as model instances.
            voting (str): 'soft' to average predicted probabilities, 'hard' for majority vote.
        """
        self.models = models
        self.voting = voting

    def fit(self, X_train, y_train):
        """
        Fit each model in the ensemble on the training data.
        
        Args:
            X_train: Training features.
            y_train: Training target.
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.train(X_train, y_train)

    def predict(self, X_test):
        """
        Predict class labels using the ensemble.
        For soft voting, average probabilities and apply a 0.5 threshold.
        For hard voting, perform majority vote.
        
        Args:
            X_test: Test features.
        
        Returns:
            np.array: Predicted class labels.
        """
        if self.voting == 'soft':
            probas = []
            for name, model in self.models.items():
                print(f"Predicting probabilities for {name}...")
                probs = model.model.predict_proba(X_test)[:, 1]
                probas.append(probs)
            avg_probas = np.mean(np.vstack(probas), axis=0)
            return (avg_probas >= 0.5).astype(int)
        elif self.voting == 'hard':
            predictions = []
            for name, model in self.models.items():
                print(f"Predicting class labels for {name}...")
                preds = model.model.predict(X_test)
                predictions.append(preds)
            predictions = np.vstack(predictions)
            # Majority vote: if sum > half the number of models, vote 1, otherwise 0.
            maj_vote = np.apply_along_axis(lambda x: int(np.sum(x) > (len(x) / 2)), axis=0, arr=predictions)
            return maj_vote
        else:
            raise ValueError("Voting must be 'soft' or 'hard'.")

    def predict_proba(self, X_test):
        """
        For soft voting: Return the averaged probability predictions.
        
        Args:
            X_test: Test features.
        
        Returns:
            np.array: Averaged predicted probabilities.
        """
        if self.voting != 'soft':
            raise ValueError("predict_proba is only available for soft voting.")
        probas = []
        for name, model in self.models.items():
            probs = model.model.predict_proba(X_test)[:, 1]
            probas.append(probs)
        avg_probas = np.mean(np.vstack(probas), axis=0)
        return avg_probas

def filter_models_by_threshold(reporter, threshold):
    """
    Filter out models that do not meet a performance threshold (e.g., minimum F1 score).
    
    Args:
        reporter: Reporter object with a dictionary 'results' storing model performance metrics.
        threshold (float): Minimum F1 score required for inclusion.
    
    Returns:
        list: List of model names (without the " (Test)" suffix) that meet the threshold.
    """
    selected = []
    for key, metrics in reporter.results.items():
        # Only consider test set metrics; keys should end with " (Test)"
        if " (Test)" in key:
            f1 = metrics.get("f1_score", 0)
            if f1 >= threshold:
                model_name = key.replace(" (Test)", "")
                selected.append(model_name)
                print(f"Including {key} with F1 score: {f1}")
            else:
                print(f"Excluding {key} with F1 score: {f1}")
    return selected
