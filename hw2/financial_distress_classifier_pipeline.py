from pathlib import Path

from sklearn.linear_model import LogisticRegression

from data_transformation import DataTransformation, binarize, discretize
from pipeline import Pipeline


def main():
    input_path = Path("./input/credit-data.csv")

    pipeline = Pipeline(input_path, 
        summarize=True,
        data_preprocessors=[],
        feature_generators=[],
        model=None,
        evaluator=None)

    pipeline.run()


if __name__ == "__main__":
    main()
