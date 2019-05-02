from datetime import timedelta
from itertools import cycle
from pathlib import Path
from types import MethodType

import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.core import Pipeline
from pipeline.transformation import (Transformation, binarize, categorize,
                                     replace_missing_with_value)

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

funded_in_60_days = Transformation("funded_in_60_days", ["date_posted", "datefullyfunded"], 
    lambda cols: (cols
        .apply(pd.to_datetime)
        .apply(lambda df: (df[1] - df[0]) <= timedelta(days=60), axis=1)
        .apply(int))
)

month_posted = Transformation("month_posted", ["date_posted"], 
    lambda cols: cols.apply(pd.to_datetime).apply(lambda _:_.dt.month)
)

def clean(dst):
    input_path = Path("./input/projects_2012_2013.csv")

    to_drop = [
        "projectid",
        "teacher_acctid",
        "schoolid",
        "school_ncesid",
        "school_latitude",
        "school_longitude",
        "school_city",
        "school_state",
        "school_metro",
        "school_district",
        "school_county",
        "school_charter",
        "school_magnet",
        "teacher_prefix",
        "primary_focus_subject",
        "primary_focus_area",
        "secondary_focus_subject",
        "secondary_focus_area",
        "resource_type",
        "poverty_level",
        "grade_level",
        "students_reached", 
        "eligible_double_your_impact_match",
        "date_posted",
        "datefullyfunded"
    ]

    # set up data cleaning pipeline
    pipeline = Pipeline(input_path, "funded_in_60_days",
        summarize=True,
        data_preprocessors=[
            categorize("school_city"),
            categorize("school_state"),
            categorize("school_metro"),
            categorize("primary_focus_subject"),
            categorize("primary_focus_area"),
            categorize("resource_type"),
            categorize("poverty_level"),
            categorize("grade_level"),
            binarize("school_charter", true_value="t"),
            binarize("school_magnet", true_value="t"),
            replace_missing_with_value("students_reached", 0),
            binarize("eligible_double_your_impact_match", true_value="t"),
            month_posted,
            funded_in_60_days
        ],
        model=None,
        name="donors-choose-preprocessing", 
        output_root_dir="output")

    # attach custom summary function
    # pipeline.summarize_data = MethodType(summarize_dc_data, pipeline)

    # run pipeline
    pipeline.run()
    pipeline.dataframe.drop(columns=to_drop).to_csv(dst)

def evaluate_models(src):
    models = { 
        "logistic-regression"    : LogisticRegression(solver="lbfgs"),
        "knn-k3"                 : KNeighborsClassifier(n_neighbors=3),
        "knn-k15"                : KNeighborsClassifier(n_neighbors=15),
        "knn-k100"               : KNeighborsClassifier(n_neighbors=100),
        "decision-tree-gini"     : DecisionTreeClassifier(criterion="gini"), 
        "decision-tree-entropy"  : DecisionTreeClassifier(criterion="entropy"), 
        "svm-linear"             : SVC(kernel="linear"), 
        "svm-rbf"                : SVC(kernel="rbf"), 
        "random-forest"          : RandomForestClassifier(),
        "boost-alpha0.1"         : GradientBoostingClassifier(learning_rate=0.1),
        "boost-alpha0.5"         : GradientBoostingClassifier(learning_rate=0.5),
        "boost-alpha2.0"         : GradientBoostingClassifier(learning_rate=2.0),
        "bagging-sample-frac0.1" : BaggingClassifier(max_samples=0.1),
        "bagging-sample-frac0.5" : BaggingClassifier(max_samples=0.5),
        "bagging-sample-frac1.0" : BaggingClassifier(max_samples=1.0)
    }

    def model_parametrized_pipeline(description, model):
        return Pipeline(src, "funded_in_60_days", 
            name="donors-choose-" + description, 
            model=model, 
            output_root_dir="output")

    for (description, model) in models.items():
        model_parametrized_pipeline(description, model).run()

def main():
    cleaned_data_path = Path("./input/clean.csv")
    if not cleaned_data_path.exists():
        clean(dst=cleaned_data_path)
    evaluate_models(src=cleaned_data_path)
    
if __name__ == "__main__":
    main()
