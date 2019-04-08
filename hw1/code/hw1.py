#!python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib2tikz import save as tikz_save
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

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
    # number of crimes reported per year
    crime_stats["count"] = 1
    counts = pd.pivot_table(crime_stats, index="Year", values="count", aggfunc=len, margins=True)
    counts["count"]["All"]/=2
    print(counts.T)
    print(counts.T.to_latex())

    # pivot table of percentages of arrest or domestic crimes 
    per_year_categories = pd.pivot_table(crime_stats, index="Year", values=["Arrest", "Domestic"], margins=True)
    per_year_percentages = 100 * np.round(per_year_categories, 4)
    print(per_year_percentages.T)
    print(per_year_percentages.T.to_latex())

    plt.figure(0)
    top_types_17  = crime_stats[crime_stats["Year"]==2017]["Primary Type"].apply(str.lower).value_counts()[:10].iloc[::-1].plot.barh(color="#285180", legend=None)
    
    plt.figure(1)
    top_types_18  = crime_stats[crime_stats["Year"]==2018]["Primary Type"].apply(str.lower).value_counts()[:10].iloc[::-1].plot.barh(color="#ff6663", legend=None)

    plt.figure(2)
    top_types_all = crime_stats["Primary Type"].apply(str.lower).value_counts()[:10].iloc[::-1].plot.barh(color="#3d3b30", legend=None)

    for (fignum, (label, chart)) in enumerate([
            ("2017", top_types_17), 
            ("2018", top_types_18),
            ("2017-18", top_types_all)]):
        plt.figure(fignum)
        plt.title(r'\printsection{\MakeUppercase{Top Crime Types in Chicago (' + label + r')}}')
        plt.xlabel(r'Number of Reports')
        plt.ylabel(r'Crime Type')
        
        # tikz_save("../latex/" + label + ".tex", figurewidth = "2.5in", figureheight = "2.5in")
    # plt.show()

# def geocode_row(row):
#     url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x={}&y={}&benchmark=4&vintage=410&format=json".format(row["Longitude"], row["Latitude"])
#     response = json.loads(requests.get(url).content)
#     geographies = response["result"]["geographies"]
#     return (geographies["Census Blocks"][0]["BLOCK"], geographies["Census Tracts"][0]["NAME"])

# def geocode(crime_stats):
#     geocoded_data_path = Path("./geocoded_crime_stats.csv")
#     if not geocoded_data_path.exists():
#         print("geocoding points")
#         # restrict geocoding to valid locations
#         crime_stats = crime_stats[crime_stats["Location"].notna()]
#         crime_stats["census_block_data"] = crime_stats.progress_apply(geocode_row, axis=1)
#         crime_stats.to_csv(geocoded_data_path)
#         return crime_stats
#     else: 
#         return pd.read_csv(geocoded_data_path)

def assign_census_tracts(crime_stats):
    boundary_shp = "./Boundaries - Census Blocks - 2000/geo_export_8e9f6d85-3c5b-429f-b625-25afcc3dea85.shp"
    census_tracts = gpd.read_file(boundary_shp).drop(columns=['perimeter', 'shape_area', 'shape_len'])
    # restrict geocoding to valid locations
    crime_stats = crime_stats[crime_stats["Location"].notna()]
    crime_stats["geometry"] = crime_stats.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis = 1)
    return gpd.tools.sjoin(gpd.GeoDataFrame(crime_stats), census_tracts, how="inner")

def analyze_ward_43(crime_stats):
    crime_stats = crime_stats[crime_stats["Ward"] == 43]
    time_period_1

if __name__ == "__main__":
    tqdm.pandas()
    crime_stats = get_cdp_data()
    summarize(crime_stats)
    geo_crime_stats = assign_census_tracts(crime_stats)

