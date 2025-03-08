import pandas as pd

class Reporter:
    def __init__(self):
        """
        Initialize the Reporter to aggregate evaluation metrics for all models.
        """
        self.results = {}

    def add_result(self, model_name, metrics):
        """
        Add model evaluation metrics.
        Args:
          - model_name (str): Identifier for the model.
          - metrics (dict): Evaluation metrics dictionary.
        """
        self.results[model_name] = metrics

    def generate_report(self, output_file="model_report.csv"):
        """
        Generate a CSV report summarizing performance metrics for each model.
        Returns:
            df_report (DataFrame): A DataFrame containing all results.
        """
        df_report = pd.DataFrame(self.results).T
        df_report.to_csv(output_file)
        print("Report generated and saved to", output_file)
        return df_report
