from pathlib import Path
from types import MethodType

import numpy as np
from sklearn.linear_model import LogisticRegression

from data_transformation import DataTransformation, replace_missing
from pipeline import Pipeline

log_monthly_income = DataTransformation("log-monthly-income", ["MonthlyIncome_clean"], "logMonthlyIncome", 
    lambda dataframe: dataframe["MonthlyIncome_clean"].apply(np.log)
)

any_late_payments = DataTransformation("any-late-payments", [""], "any-late-payments", None)

def summarize_financial_distress_input(self, path):
    self.logger.info("Running custom summary function")
    return self

def main():
    input_path = Path("./input/credit-data.csv")

    # set up pipeline
    pipeline = Pipeline(input_path, 
        summarize=True,
        data_preprocessors=[
            replace_missing("MonthlyIncome"), 
            replace_missing("NumberOfDependents"),
            log_monthly_income],
        feature_generators=[],
        model=None,
        name="financial-distress-classifier", 
        output_root_directory="output")

    # attach custom summary function
    # pipeline.summarize_data = MethodType(summarize_financial_distress_input, pipeline)

    # let's go classify things
    pipeline.run()

    return pipeline

if __name__ == "__main__":
    main()
