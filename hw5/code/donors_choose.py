import datetime
from itertools import cycle
from pathlib import Path

import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from pipeline.core import Pipeline
from pipeline.grid import Grid
from pipeline.splitter import Splitter
from pipeline.transformation import (Transformation, binarize, categorize,
                                     replace_missing_with_value, scale_by_max,
                                     to_datetime)
from types import MethodType
tqdm.pandas()

colors = cycle(["#0f5257", "#0b3142", "#9c92a3", "#734b5e", "#92afd7"])

def summarize_donors_choose_data(self):
    self.logger.info("Running custom summary function")
    df = self.dataframe
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

# data cleaning and transformation
funded_in_60_days = Transformation("funded_in_60_days", ["date_posted", "datefullyfunded"],
    lambda cols: cols.progress_apply(lambda df: int((df[1] - df[0]) >= datetime.timedelta(days=60)), axis=1))

month_posted = Transformation("month_posted", ["date_posted"],
    lambda cols: cols.apply(pd.to_datetime).apply(lambda _:_.dt.month))

def main(config_path):
    script_dir = Path(__file__).parent
    with open(script_dir/config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    pipeline_name = config["pipeline"]["name"]
    input_path    = script_dir/config["data"]["input_path"]
    output_dir    = script_dir/config["data"]["output_dir"]
    target        = config["pipeline"]["target"]

    splitter   = Splitter.from_config(config["pipeline"]["test_train"])
    model_grid = Grid.from_config(config["models"])

    pipeline = Pipeline(
        name               = pipeline_name,
        input_source       = input_path,
        target             = target,
        summarize          = True,
        splitter           = splitter,
        model_grid         = model_grid,
        output_root_dir    = output_dir,
        verbose            = True,
        positive_label     = 1,
        data_cleaning      = [
            to_datetime("date_posted"),
            to_datetime("datefullyfunded"),
            month_posted,
            funded_in_60_days],
        feature_generators = [
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
        ])
    
    # set custome summary function
    pipeline.summarize_data = MethodType(summarize_donors_choose_data, pipeline)
    
    # run pipeline
    pipeline.run()

if __name__ == "__main__":
    main("config.yml")
