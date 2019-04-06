#!python3

from pathlib import Path

import pandas as pd
import requests


def get_cdp_data():
    # download crime data if we don't have it locally
    base_url = "https://data.cityofchicago.org/api/views/{}/rows.csv?accessType=DOWNLOAD"
    crime_resources = { 
        2017: (Path("./crime_data_2017.csv"), "3i3m-jwuy"),
        2018: (Path("./crime_data_2018.csv"), "d62x-nvdr"),
    }

    for (year, (path, identifier)) in crime_resources.items():
        if not path.exists():
            url = base_url.format(identifier)
            print("{} data not found locally, downloading from {}".format(year, url))
            response = requests.get(url)
            with path.open("wb") as f:
                f.write(response.content)

    crime_stats = pd.concat([
        pd.read_csv(crime_resources[2017][0]), 
        pd.read_csv(crime_resources[2018][0])
    ])

    return crime_stats

def summarize(crime_stats):
    pass

if __name__ == "__main__":
    crime_stats = get_cdp_data()
    summarize(crime_stats)

