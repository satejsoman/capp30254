#!python3

import datetime
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from census import Census
from matplotlib2tikz import save as tikz_save
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

def summarize_changes(crime_stats):
    # overall changes
    print(crime_stats.groupby("Year").size().to_frame().T.to_latex())
    print((100 * crime_stats.groupby("Year").size().pct_change()).to_frame().T.to_latex())

    # changes per type
    # per_type_changes = 100 * crime_stats.groupby(["Primary Type", "Year"]).size().unstack().T.pct_change().stack().sort_values(ascending=False)
    # print(per_type_changes.to_latex())

def assign_census_tracts(crime_stats):
    boundary_shp = "./Boundaries - Census Blocks - 2000/geo_export_8e9f6d85-3c5b-429f-b625-25afcc3dea85.shp"
    census_tracts = gpd.read_file(boundary_shp).drop(columns=['perimeter', 'shape_area', 'shape_len'])
    # restrict geocoding to valid locations
    crime_stats = crime_stats[crime_stats["Location"].notna()]
    crime_stats["geometry"] = crime_stats.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis = 1)
    return gpd.tools.sjoin(gpd.GeoDataFrame(crime_stats), census_tracts, how="inner")

def analyze_demographic_data(crime_stats, census_client):
    tract_numbers = set(crime_stats["census_tra"].to_list())
    illinois = '17'
    cook_county = '031'
    acs_vars = {
        'NAME' : "tract_name",
        'B01003_001E': 'total_pop',
        'B02001_003E': 'black_pop',
        'B03003_003E': 'hispanic_pop',
        'B19013_001E': 'median_income',
        'B22002_001E': 'child_snap'
    }
    response = census_client.acs5.state_county_tract(list(acs_vars.keys()), illinois, cook_county, Census.ALL)
    demography = pd.DataFrame([elem for elem in response if elem["tract"] in tract_numbers]).rename(columns=acs_vars)
    # normalize by population
    demography[["black_pct", "hispanic_pct", "child_snap_pct"]] = demography[["black_pop", "hispanic_pop", "child_snap"]].div(demography.total_pop, axis=0)
    # deal with bottom-coded values for income
    demography["median_income"][demography["median_income"] == demography["median_income"].min()] = None

    demography["tract"] = pd.to_numeric(demography["tract"])
    demography.set_index("tract")
    crime_stats["census_tra"] = pd.to_numeric(crime_stats["census_tra"])
    crime_stats = crime_stats.merge(demography, left_on=["census_tra"], right_on=["tract"])
    
    demographic_vars = ["black_pct", "hispanic_pct", "child_snap_pct", "median_income"]

    print("battery")
    print(crime_stats[crime_stats["Primary Type"] == "BATTERY"][demographic_vars].describe().to_latex())
    print("homicide")
    print(crime_stats[crime_stats["Primary Type"] == "HOMICIDE"][demographic_vars].describe().to_latex())

    print("homicide over time")
    print(crime_stats[(crime_stats["Primary Type"] == "HOMICIDE") & (crime_stats["Year"] == 2017)][demographic_vars].describe().to_latex())
    print(crime_stats[(crime_stats["Primary Type"] == "HOMICIDE") & (crime_stats["Year"] == 2018)][demographic_vars].describe().to_latex())

    print("deceptive practice vs. sex offense")
    print(crime_stats[crime_stats["Primary Type"] == "DECEPTIVE PRACTICE"][demographic_vars].describe().to_latex())
    print(crime_stats[crime_stats["Primary Type"] == "SEX OFFENSE"][demographic_vars].describe().to_latex())

def analyze_ward(crime_stats, ward=43):
    # "All told, crime rose 16 percent in the same 28-day time period in just one year"
    one_month = datetime.timedelta(days = 28) 
    target_2017 = datetime.datetime(year=2017, month=7, day=26)
    target_2018 = datetime.datetime(year=2018, month=7, day=26)
    preceding_month_2017 = target_2017 - one_month
    preceding_month_2018 = target_2018 - one_month

    crime_stats = crime_stats[crime_stats["Ward"] == ward]
    crime_stats["datetime"] = pd.to_datetime(crime_stats["Date"])
    crime_stats = crime_stats[
        ((preceding_month_2017 <= crime_stats["datetime"]) & (crime_stats["datetime"] <= target_2017)) | 
        ((preceding_month_2018 <= crime_stats["datetime"]) & (crime_stats["datetime"] <= target_2018))
    ]
    
    print(crime_stats.groupby("Year").size())
    print(crime_stats.groupby("Year").size().pct_change())
    print(crime_stats.groupby("Year").size().to_latex())
    print(crime_stats.groupby("Year").size().pct_change().to_latex())

    crime_stats = crime_stats[crime_stats["Primary Type"].isin(["ROBBERY", "BATTERY", "BURGLARY", "MOTOR VEHICLE THEFT"])]
    crime_agg_2017 = crime_stats[crime_stats["Year"] == 2017]["Primary Type"].value_counts()
    crime_agg_2018 = crime_stats[crime_stats["Year"] == 2018]["Primary Type"].value_counts()
    crime_agg_2017.name = "2017"
    crime_agg_2018.name = "2018"
    crime_agg = pd.DataFrame([crime_agg_2017, crime_agg_2018])
    print(crime_agg.T)
    print(100*crime_agg.pct_change().T)
    print(crime_agg.T.to_latex())
    print((100*crime_agg.pct_change().T).to_latex())

def analyze_ward_to_date(crime_stats, ward=43, target_month=7, target_day=26):
    year_start   = lambda year: datetime.datetime(year=year, month=1, day=1)
    year_to_date = lambda year: datetime.datetime(year=year, month=target_month, day=target_day)
    crime_stats = crime_stats[crime_stats["Ward"] == ward]
    crime_stats["Date"] = pd.to_datetime(crime_stats["Date"])
    crime_stats = crime_stats[
        ((year_start(2017) <= crime_stats["Date"]) & (crime_stats["Date"] <= year_to_date(2017))) | 
        ((year_start(2018) <= crime_stats["Date"]) & (crime_stats["Date"] <= year_to_date(2018)))
    ]

    print(crime_stats.groupby("Year").size())
    print(100 * crime_stats.groupby("Year").size().pct_change())

def analyze_crime_for_block(crime_stats, block_address):
    return 100 * crime_stats[crime_stats["Block"].str.contains("021XX S MICHIGAN")]["Primary Type"].value_counts(normalize=True)

def theft_probabilities(crime_stats, areas):
    return 100 * crime_stats[crime_stats["Primary Type"] == "THEFT"]["Community Area"].value_counts(normalize=True)[[float(a) for a in areas]]

if __name__ == "__main__":
    crime_stats = get_cdp_data()
    
    # summarize(crime_stats)
 
    # summarize_changes(crime_stats)

    # geo_crime_stats = assign_census_tracts(crime_stats)
    
    # census_client = Census(open("censuskey").read().strip())
    
    # analyze_demographic_data(geo_crime_stats, census_client)
    
    # analyze_crime_for_block(crime_stats, "021XX S MICHIGAN")
    
    # analyze_ward(crime_stats, ward=43)

    analyze_ward_to_date(crime_stats, ward=43, target_month=7, target_day=26)
    analyze_ward_to_date(crime_stats, ward=43, target_month=12, target_day=31)
