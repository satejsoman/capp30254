from pathlib import Path

# from sklearn.linear_model import LogisticRegression

from data_transformation import DataTransformation, binarize, discretize, replace_missing
from pipeline import Pipeline

def main():
    input_path = Path("./input/credit-data.csv")

    pipeline = Pipeline(input_path, 
        summarize=True,
        data_preprocessors=[
            replace_missing("MonthlyIncome"), 
            replace_missing("NumberOfDependents")],
        feature_generators=[],
        model=None,
        evaluator=None,
        name="financial-distress-classifier", 
        output_root_directory="output")

    pipeline.run()

    return pipeline

if __name__ == "__main__":
    main()