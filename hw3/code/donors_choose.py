import datetime
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
                                     replace_missing_with_value, scale_by_max)

colors = cycle(["#0f5257", "#0b3142", "#9c92a3", "#734b5e", "#92afd7"])

# custom pipeline stages 
def load_from_dataframe(df):
    def _load(self):
        self.logger.info("Loading data from existing data frame")
        self.dataframe = df
        return self
    return _load

def summarize_dc_data(self):
    self.logger.info("Running custom summary function")
    df = self.dataframe.drop(columns = ["school_ncesid"])
    summary = df.describe(percentiles=[]).drop("count").T

    with (self.output_dir/"summary.tex").open('w') as fer:
        summary.to_latex(buf=fer, float_format="%.2f")

    with (self.output_dir/"missing.tex").open('w') as fer:
        pd.DataFrame([df.isnull().sum()]).T.to_latex(buf=fer, float_format="%.2f")

    for (column, color) in zip(df.columns, colors):
        try: 
            df[[column]].plot.hist(facecolor=color, legend=None, bins=20)
            plt.gca().legend().remove()
            matplotlib2tikz.save(self.output_dir/(column + ".tex"), figureheight="3in", figurewidth="3in")
        except TypeError:
            pass
    return self

def temporal_test_train_split(df):
    end_date = datetime.datetime(year=2013, month=12, day=31)
    six_months = datetime.timedelta(days=180)

    split_dates = [end_date - (k * six_months) for k in (3, 2, 1)]

    df["date_posted"] = pd.to_datetime(df["date_posted"])

    train_set_indices, test_set_indices = [], []

    for split_date in split_dates:
        train_set_indices.append(df["date_posted"] < split_date)
        test_set_indices.append((df["date_posted"] >= split_date) & (df["date_posted"] < (split_date + six_months)))
    
    def _split(self):
        self.logger.info("Creating train/test data sets")
        for (train_index, test_index) in zip(train_set_indices, test_set_indices):
            self.training_sets.append({
                "X": self.dataframe.loc[train_index, self.features], 
                "y": self.dataframe.loc[train_index, self.target]
            })
            self.testing_sets.append({
                "X": self.dataframe.loc[test_index, self.features], 
                "y": self.dataframe.loc[test_index, self.target]
            })

        return self
    return _split

# data cleaning and transformation
funded_in_60_days = Transformation("funded_in_60_days", ["date_posted", "datefullyfunded"], 
    lambda cols: (cols
        .apply(pd.to_datetime)
        .apply(lambda df: (df[1] - df[0]) <= datetime.timedelta(days=60), axis=1)
        .apply(int)))

month_posted = Transformation("month_posted", ["date_posted"], 
    lambda cols: cols.apply(pd.to_datetime).apply(lambda _:_.dt.month))

# pipelines 
def explore(src, dst):
    target = "funded_in_60_days"
    df = pd.read_csv(src)
    df[target] = funded_in_60_days(df[funded_in_60_days.input_column_names])

    pipeline = Pipeline(src, target, summarize=True, 
        name="1-donors-choose-exploration", 
        output_root_dir="output")

    pipeline.load_data = MethodType(load_from_dataframe(df), pipeline)
    pipeline.summarize_data = MethodType(summarize_dc_data, pipeline)

    pipeline.run()
    pipeline.dataframe.to_csv(dst)

def clean(src, dst):
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
        "datefullyfunded",
        "students_reached_clean",
        "total_price_including_optional_support"
    ]

    # set up data cleaning pipeline
    pipeline = Pipeline(src, "funded_in_60_days",
        data_preprocessors=[
            categorize("school_city"),
            categorize("school_state"),
            categorize("primary_focus_subject"),
            categorize("primary_focus_area"),
            categorize("resource_type"),
            categorize("poverty_level"),
            categorize("grade_level"),
            binarize("school_charter", true_value="t"),
            binarize("school_magnet", true_value="t"),
            replace_missing_with_value("students_reached", 0),
            scale_by_max("students_reached_clean"),
            scale_by_max("total_price_including_optional_support"),
            binarize("eligible_double_your_impact_match", true_value="t"),
            month_posted,
            funded_in_60_days
        ],
        model=None,
        name="2-donors-choose-preprocessing", 
        output_root_dir="output")

    # run pipeline
    pipeline.run()
    pipeline.dataframe.drop(columns=to_drop).to_csv(dst)

def evaluate_models(original, src):
    df = pd.read_csv(original)

    _models = {}
    model_grid = [
        {"log-reg-c{}-penalty{}".format(c,p) : LogisticRegression(solver="lbfgs", C=c, penalty=p) for c in (10**-2, 10**-1, 1 , 10, 10**2) for p in ('l1', 'l2')},
        {"knn-k{}".format(k)                 : KNeighborsClassifier(n_neighbors=k)                for k in (10, 50, 100)},
        {"decision-tree-{}".format(c)        : DecisionTreeClassifier(criterion=c)                for c in ("gini", "entropy")},
        {"boost-alpha{}".format(a)           : GradientBoostingClassifier(learning_rate=a)        for a in (0.1, 0.5, 2.0)},
        {"bagging-sample-frac{}".format(f)   : BaggingClassifier(max_samples=f)                   for f in (0.1, 0.5, 1.0)},
        {"random-forest"                     : RandomForestClassifier()},
    ]
    for classifier in model_grid:
        _models.update(classifier)
    
    models = { 
        "logistic-regression"    : LogisticRegression(solver="lbfgs"),
        # "knn-k3"                 : KNeighborsClassifier(n_neighbors=3),
        # "knn-k15"                : KNeighborsClassifier(n_neighbors=15),
        # "knn-k100"               : KNeighborsClassifier(n_neighbors=100),
        # "decision-tree-gini"     : DecisionTreeClassifier(criterion="gini"), 
        # "decision-tree-entropy"  : DecisionTreeClassifier(criterion="entropy"), 
        # "random-forest"          : RandomForestClassifier(),
        # "boost-alpha0.1"         : GradientBoostingClassifier(learning_rate=0.1),
        # "boost-alpha0.5"         : GradientBoostingClassifier(learning_rate=0.5),
        # "boost-alpha2.0"         : GradientBoostingClassifier(learning_rate=2.0),
        # "bagging-sample-frac0.1" : BaggingClassifier(max_samples=0.1),
        # "bagging-sample-frac0.5" : BaggingClassifier(max_samples=0.5),
        # "bagging-sample-frac1.0" : BaggingClassifier(max_samples=1.0),
        # "svm-linear"             : SVC(kernel="linear", gamma="scale", verbose=True), 
        # "svm-rbf"                : SVC(kernel="rbf", gamma="scale", verbose=True), 
    }

    def model_parametrized_pipeline(description, model):
        return Pipeline(src, "funded_in_60_days", 
            name="3-donors-choose-classifier-" + description, 
            model=model, 
            output_root_dir="output")

    evaluations = []
    for (description, model) in models.items():
        pipeline = model_parametrized_pipeline(description, model)
        pipeline.generate_test_train = MethodType(temporal_test_train_split(df), pipeline)
        pipeline.run()
        evaluations.append(pipeline.model_evaluations)

def main():
    input_path = Path("./input/projects_2012_2013.csv")
    xplor_path = Path("./input/explored.csv")
    clean_path = Path("./input/clean.csv")
    
    if not xplor_path.exists():
        explore(src=input_path, dst=xplor_path)

    if not clean_path.exists():
        clean(src=xplor_path, dst=clean_path)
    
    evaluate_models(original=input_path, src=clean_path)
    
if __name__ == "__main__":
    main()
