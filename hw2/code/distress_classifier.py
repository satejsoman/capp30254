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

colors = cycle(["#d5d6aa", "#9dbbae", "#769fb6", "#188fa7"])

# categorical variable
age_decade = Transformation("age-decade", ["age"], "age_decade", 
    lambda col: col.apply(lambda x: 10*(x//10))
)

# binary variable
any_late_payments = Transformation("any-late-payments", ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'], "any_late_payments", 
    lambda cols: cols.apply(lambda x: int(np.sum(x) > 0), axis=1)
)

log_monthly_income = Transformation("log-monthly-income", ["MonthlyIncome_clean"], "log_monthly_income", 
    lambda col: col.apply(lambda x: np.log(x + 1e-10))
)

def summarize_fd_data(self):
    self.logger.info("Running custom summary function")
    df = self.dataframe
    summary = (df
        .describe(percentiles=[])
        .drop("count")
        .drop("50%")
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

def main():
    input_path = Path("./input/credit-data.csv")

    # set up pipeline
    pipeline = Pipeline(input_path, "SeriousDlqin2yrs",
        summarize=True,
        data_preprocessors=[
            replace_missing("MonthlyIncome"), 
            replace_missing("NumberOfDependents")],
        feature_generators=[
            age_decade,
            any_late_payments,
            log_monthly_income],
        model=LogisticRegression(solver="lbfgs"),
        name="financial-distress-classifier", 
        output_root_dir="output")

    # attach custom summary function
    pipeline.summarize_data = MethodType(summarize_fd_data, pipeline)

    # run pipeline
    pipeline.run()

    return pipeline

if __name__ == "__main__":
    main()
