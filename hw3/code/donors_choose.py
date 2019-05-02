from itertools import cycle
from pathlib import Path
from types import MethodType

import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline.core import Pipeline
from pipeline.transformation import Transformation, replace_missing

colors = cycle([])

def summarize_dc_data(self):
    self.logger.info("Running custom summary function")
    df = self.dataframe
    summary = (df
        .describe(percentiles=[])
        .drop("count")
        .append(pd.DataFrame([df.corr()[self.target].apply(np.abs), df.isnull().sum()], index=["abs corr", "missing"]))
        .T)
    summary.rename(dict(zip(summary.index, [
            "id", 
            "delinquency",
            "revolving utilization",
            "age",
            "zipcode",
            "payments 30-59 days late",
            "debt ratio",
            "monthly income",
            "number of credit lines",
            "payments 90 days late",
            "number of real estate loans",
            "payments 60-89 days late",
            "number of dependents"])), inplace=True)
    summary.sort_values("abs corr", inplace=True, ascending=False)

    with (self.output_dir/"summary.tex").open('w') as fer:
        summary.to_latex(buf=fer, float_format="%.2f")

    for (column, color) in zip(df.columns, colors):
        df[[column]].plot.hist(facecolor=color, legend=None, bins=20)
        plt.gca().legend().remove()
        matplotlib2tikz.save(self.output_dir/(column + ".tex"), figureheight="3in", figurewidth="3in")
    return self

funded_in_60_days = Transformation("funded_in_60_days", 
    [], 
    lambda cols: None
)

def clean(dst):
    input_path = Path("./input/projects_2012_2013.csv")

    donors_choose_preprocessors = [
        ...
    ]

    # set up data cleaning pipeline
    pipeline = Pipeline(input_path, "funded_in_60_days",
        summarize=True,
        data_preprocessors=donors_choose_preprocessors,
        model=None,
        name="donors-choose-preprocessing", 
        output_root_dir="output")

    # attach custom summary function
    # pipeline.summarize_data = MethodType(summarize_dc_data, pipeline)

    # run pipeline
    pipeline.run()
    pipeline.dataframe.to_csv(dst)

def evaluate_models(src):
    donors_choose_features = []

    models_to_run = { 
        ... : ...
    }

    def model_parametrized_pipeline(description, model):
        return Pipeline(src, "funded_in_60_days",
            summarize=False,
            features=donors_choose_features,
            name="donors-choose-" + description,
            model=model,
            output_root_dir="output")

    for (description, model) in models_to_run.items():
        model_parametrized_pipeline(description, model).run()

def main():
    cleaned_data_path = Path("./input/projects_2012_2013.csv")
    if not cleaned_data_path.exists():
        clean(dst=cleaned_data_path)
    evaluate_models(src=cleaned_data_path)
    
if __name__ == "__main__":
    main()
