import datetime

import pandas as pd

from pipeline.transformation import Transformation

# data cleaning and transformation
funded_in_60_days = Transformation("funded_in_60_days", ["date_posted", "datefullyfunded"],
    lambda cols: cols.progress_apply(lambda df: int((df[1] - df[0]) >= datetime.timedelta(days=60)), axis=1))

month_posted = Transformation("month_posted", ["date_posted"],
    lambda cols: cols.apply(pd.to_datetime).apply(lambda _:_.dt.month))
