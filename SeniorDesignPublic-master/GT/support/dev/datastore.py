import re
import time
import requests
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import censusdata

class DataStore():
    """Cleans and holds all data necessary for beer selection tool.

    This class is used to load and clean data for use in the regression models
    for demand forecasting and BeerValueEstimator (MTOP)

    Attributes:
        raw_sales (pd.df): Daily raw sales data on tapped beer
        raw_sports (pd.df): Daily number of games for a league

        sales (pd.df): Clean aggregated sales data
        sports(pd.df): Clean aggregated sports data
        timeframe (str): The time frame of aggregation (Week, Month)

        costs (pd.df): Clean cost data

        demo (pd.df): Clean demographic and location data
        beer_styles (pd.df): Mapping segment to beer style

        exog (pd.df): Final dataframe of exogenous variables
        base_plan (list) : List of the segments in the baseplan
        base_exog (pd.df): Final dataframe of exogenous variables for base plan beers
        style_exog (pd.df): Final dataframe of exogenous variables for all other beers
        style_exog_avg (pd.df): Final dataframe of exogenous variables for all other beers averaged
    """

    def __init__(self, **kwargs):
        """This is where we load all the necessary data that needs to be cleaned.

        Args:
            **kwargs (dict): For future inputs
        """
        # Raw Dataframes hold the cleaned data with minimal aggregation
        # These track daily sales and sports events
        self.raw_sales = pd.DataFrame()
        self.raw_sports = pd.DataFrame()
        
        # Cost data frames hold the cost per ounce for each beer segment
        # Costs Mapped Manually
        self.costs = pd.DataFrame()
        
        # Demo and store_info hold the demographic and restaurant information
        # Beer styles holds the manually mapped style for each beer segment
        # Demo and Beer style mappings are not time sensitive
        self.demo = pd.DataFrame()
        self.store_info = pd.DataFrame()
        self.beer_styles = pd.DataFrame()

        # The final sales and sports dfs are populated by calling 
        # group_<sports/sales>() or the helper method group_data()
        # timeframe holds a string to indicate what grouping is stored
        # timeframe = month --> Sales and sports are aggregated on a monthly basis
        self.time = ''
        self.sales = pd.DataFrame()
        self.sports = pd.DataFrame()

        # The exog files contain the final cleaned and merged data
        # base_exog contains segment level data for the beers 
        # determined to be on the base plan
        # style_exog contains style averages of sales and price
        self.exog = pd.DataFrame()
        self.base_plan = ''
        self.base_exog = pd.DataFrame()
        self.style_exog = pd.DataFrame()
        self.style_exog_avg = pd.DataFrame()

        self.intra_portfolio = pd.DataFrame()
        self.num_styles = pd.DataFrame()

        # Storing Atlanta segment corrections
        # These are similarly named segments formatted old : new
        # These unfortunately need to be manually verified
        # See the helper method get_similar segments to generate a potential dictionary
        self.corrections = {"Angry Orchard Crisp Apple": "Angry Orchard Crisp Apple Cider",
                            "Blue Moon": "Blue Moon Belgian White",
                            "Dundee Honey Brown": "Dundee Original Honey Brown",
                            "Hop Dang Diggity": "Jekyll Hop Dang Diggity",
                            "Samuel Adams Boston 26.2 Glass": "Samuel Adams Boston 26.2",
                            "Scofflaw Dirty Beaches ": "Scofflaw Dirty Beaches",
                            "Southbound  IPA": "Southbound IPA",
                            "Sweetwater Low Hanging Fruit Passionfrui": "Sweetwater Low Hanging Fruit Passionfruit",
                            "Jack-O Traveler": "Jack-O-Traveler"
                            }

    def load_sales(self, sales_filepath=None, group=True):
        """Loads and cleans sales/POS data.

        This method will take in the pos data and output
        a cleaned version of sales data that can be useful
        for inputting into the regression model.
        Date columns, and ItemName2Vol conversion among others.

        Args:
            sales_filepath (str): file path to POS.xlsx
            group (bool): Daily aggregation on item sales
                          Help when reconciling item names and volumes
                          for big POS files. Set to False if you need 
                          raw_sales to look at every individual purhcase

        Returns:
            self.sales (pd.df) : cleaned sales data
        """
        def item_name_2_vol(item_name):
            """Given an item_name return the corresponding volume

            Args:
                item_name (str): ItemName from POS data

            Returns:
                quantity_oz (float): quantity in ounces
            """
            item_name = item_name.lower()
            quantity_oz = np.NaN

            if "reg" in item_name:
                # assuming reg beers are 16 oz pours
                quantity_oz = 16.0
            elif "tall" in item_name:
                # assuming tall beers are 23 oz pours
                quantity_oz = 23.0
            elif "smpl" in item_name:
                # assuming that samples are only recorded and
                # are not taken into account due to no sales
                quantity_oz = 1.5
            elif "bucket" in item_name or "bckt" in item_name:
                # assuming buckets have 5 standard 12oz beers
                # We want to ignore packaged beer
                # quantity_oz = 60.0
                quantity_oz = np.NaN
            elif "ptchr" in item_name or "pitcher" in item_name:
                # assumes pitchers are 60oz
                quantity_oz = 60.0
            elif "6ooz" in item_name:
                # accounts for the once time it was misspelled
                quantity_oz = 60.0
            elif "oz" in item_name:
                potential_oz = re.findall(r"\d+", item_name)
                if len(potential_oz) != 0:
                    quantity_oz = int(potential_oz[0])
                else:
                    pass
            else:
                quantity_oz = np.NaN

            return quantity_oz

        def menu_name_2_vol(menu_name):
            """Given an menu_name return the corresponding volume

            Args:
                menu_name (str): CommonMenuName from POS data

            Returns:
                quantity_oz (float): quantity in ounces
            """
            menu_name = menu_name.lower()
            quantity_oz = np.NaN

            if "pkg" in menu_name:
                # assumes pkg reffers to 12oz bottles/cans
                quantity_oz = 12.0
            elif "oz" in menu_name:
                potential_oz = re.findall(r"\d+", menu_name)
                if len(potential_oz) != 0:
                    quantity_oz = int(potential_oz[0])
                else:
                    pass
            elif "reg" in menu_name:
                # assumes reg beers are 16oz pours
                quantity_oz = 16.0
            elif "tall" in menu_name:
                # assumes tall beers are 23oz pours
                quantity_oz = 23.0
            elif "smpl" in menu_name or "sample" in menu_name:
                # assumes that samples are only recorded and
                # are not taken into account due to no sales
                quantity_oz = 1.5
            elif "bckt" in menu_name or "bucket" in menu_name:
                # assumes buckets have 5 standard 12oz beers
                # We want to ignore packaged beer
                # quantity_oz = 60.0
                quantity_oz = np.NaN
            elif "ptchr" in menu_name or "pitcher" in menu_name:
                # assumes pitchers are 60oz
                quantity_oz = 60.0
            else:
                quantity_oz = np.NaN

            return quantity_oz

        def fix_segments(segment):
            """
            Given an segment return the correct segment

            Args:
                segment (str): Segment from POS data

            Returns:
                correct_segment (str): corrected segment
            """
            # returns a segment from the corrections dictionary or 
            # returns the same segment that was passed in
            correct_segment = self.corrections.get(segment, segment)

            return correct_segment

        POS_converter = {"Segment": str,
                         "DateKey": str,
                         "FiscalWeek": str,
                         "ItemName": str,
                         "CommonMenuName": str
                         }

        if sales_filepath.endswith(".xlsx"):
            sales = pd.read_excel(sales_filepath, converters=POS_converter)
        elif sales_filepath.endswith(".csv"):
            sales = pd.read_csv(sales_filepath, converters=POS_converter)

        sales.dropna(inplace=True)

        # Preliminary daily grouping to handle BIG data
        # Necessary as ItemName and CommonMenu Name reconciliation
        # in big data files. Set group=False at your own risk
        if group:
            groups = ["RestaurantNumber",
                      "DesignatedMarketArea-TV",
                      "CheckSummarykey",
                      "PluNumber",
                      "ItemName",
                      "CommonMenuName",
                      "Segment",
                      "SubCategory",
                      "SalesCategory",
                      "DateKey",
                      "DayNameWeek",
                      "FiscalYear",
                      "FiscalWeek"
                      ]
            
            aggregates = ["DiscPric_Sales",
                          "Price_Sales",
                          "Beer_Orders"
                          ]

            sales = sales.groupby(groups, as_index=False).sum()
            sales = sales[groups + aggregates]

        # Correct similar segments
        sales["Segment"] = sales["Segment"].apply(fix_segments)

        # Format the date and time columns
        sales["DateTime"] = pd.to_datetime(sales["DateKey"], format="%Y%m%d")
        sales["Year"] = sales["DateTime"].apply(lambda x: x.year)
        sales["Month"] = sales["DateTime"].apply(lambda x: x.month)
        sales["Day"] = sales["DateTime"].apply(lambda x: x.day)

        sales["Week"] = sales["DateTime"].apply(lambda x: x.week)
        
        # Counter for Week counting up from week 1 in 2017
        sales["WeekTime"] = sales["DateTime"].apply(
            lambda x: ((x.year - 2017) * 52) + x.week)

        # Counter for Month counting up from month 1 in 2017
        sales["MonthTime"] = sales["DateTime"].apply(
            lambda x: ((x.year - 2017) * 12) + x.month)

        # Converting Volume from the ItemName and MenuName to ounces
        sales["ItemVolume"] = sales["ItemName"].apply(item_name_2_vol)
        sales["MenuVolume"] = sales["CommonMenuName"].apply(menu_name_2_vol)

        # Consolidate Volume from both volume conversions
        sales["Comparison"] = (sales["ItemVolume"] == sales["MenuVolume"])
        xor = sales[sales["ItemVolume"].notna() ^ sales["MenuVolume"].notna()]
        xor_volume = xor["ItemVolume"].fillna(0) + xor["MenuVolume"].fillna(0)
        agreed_volume = sales[sales["Comparison"] == True]["MenuVolume"]
        disagreed_volume = sales[sales["Comparison"] == False]["MenuVolume"]

        sales["VolumeOz"] = np.NaN 
        sales["VolumeOz"] = sales["VolumeOz"].fillna(xor_volume)
        sales["VolumeOz"] = sales["VolumeOz"].fillna(agreed_volume)
        sales["VolumeOz"] = sales["VolumeOz"].fillna(disagreed_volume)

        # Filter only tapped beers
        sales = sales[((sales["VolumeOz"] == 16) | 
                      (sales["VolumeOz"] == 23) | 
                      (sales["VolumeOz"] == 60)
                      )]
        sales = sales.dropna(subset=["VolumeOz"])
        # Remove Segments with oz, hypothesized to be cans
        sales = sales[~sales["Segment"].str.contains("oz ")]

        # Remove sales with errors in the number of beer orders
        sales = sales[sales['Beer_Orders'] >= 1]

        # Calculate the total volume of beer sold in ounces
        sales["TotalVolumeOz"] = sales["Beer_Orders"] * sales["VolumeOz"]

        # Concatenate new sales data to old stored sales data
        # allows loading multiple files
        self.raw_sales = pd.concat([self.raw_sales, sales],
                                    axis=0,
                                    ignore_index=True,
                                    sort=False)

        return self.raw_sales

    def get_similar_segments(self, ratio):
        """
        Return a dictionary of similarly named segments 
        to correct for incorrectly named segments.
        
        The ratio determines the how similar the segment names will be
        and the longest segment named is assumed to be the correct one
        
        Args:
            ratio (float): similarity ratio between 0 - 1 0 being completely different

        Returns:
            corrections (dict): dictionary formatted:

                corrections[incorrect] = correct
        """
        segments = self.sales["Segment"].unique()
        corrections = {}

        for item1 in segments:
            for item2 in segments[segments != item1]:
                ratio = SequenceMatcher(None, item1, item2).ratio()
                if ratio >= 0.75 and item1 != item2 and item1 not in corrections and item2 not in corrections:
                    seq = (item1, item2)
                    correct = max(enumerate(seq), key=lambda x: len(x[1]))
                    incorrect = seq[1 - correct[0]]
                    corrections[incorrect] = correct[1]

        return corrections

    def load_intra(self, inplace=True):
        """Load the intra portfolio columns. 

        These track the prices of the beers in the base plan and adds them as 
        separate columns in the exog df. It considers only the price at each 
        respective location and time.
        """
        segment_groups = ["RestaurantNumber", "Segment", "VolumeOz"]
        segments = self.exog.groupby(segment_groups, as_index=False)
        restaurants = self.exog.groupby("RestaurantNumber", as_index=False)

        if "MonthTime" in self.exog.columns:
            time = "MonthTime"
        elif "WeekTime" in self.exog.columns:
            time = "WeekTime"

        intra_portfolio = pd.DataFrame()
        for rest_num, rest_data in restaurants:
            intra_df = pd.DataFrame()
            intra_df[time] = rest_data[time].unique()
            intra_df["RestaurantNumber"] = rest_num
            for i, intra in enumerate(self.base_plan):
                intra_key = "intra_{}".format(i)
                intra_data = segments.get_group((rest_num, intra, 23))
                intra_data = intra_data[["RestaurantNumber", time, "Price"]]
                intra_data = intra_data.rename(columns={"Price":intra_key})
                intra_df = intra_df.merge(intra_data, on=["RestaurantNumber", time], how='left')

            intra_portfolio = pd.concat([intra_portfolio, intra_df], 
                                        axis=0, 
                                        ignore_index=True, 
                                        sort=False)

        intra_portfolio.fillna(0, inplace=True)
        self.intra_portfolio = intra_portfolio

        if inplace:
            self.exog = self.exog.merge(intra_portfolio, on=["RestaurantNumber", time])

        return self.intra_portfolio, ["RestaurantNumber", time]

    def load_num_styles(self, inplace=False):
        """Calculate the number of similar beer styles sold each month at each restaurant"""
        if "MonthTime" in self.exog.columns:
            time = "MonthTime"
        elif "WeekTime" in self.exog.columns:
            time = "WeekTime"

        style_groups = ["RestaurantNumber", time, "Style"]
        styles = self.exog.groupby(style_groups, as_index=False)
        
        num_styles = pd.DataFrame()
        for group_id, style_data in styles:
            rest_num, group_time, style = group_id
            style_dict = {'RestaurantNumber': [rest_num], time: [group_time], "Style": [style]}
            num_styles_df = pd.DataFrame(style_dict)
            num_styles_df["NumStyles"] = len(style_data["Segment"].unique())
            num_styles = pd.concat([num_styles, num_styles_df],
                                    axis=0,
                                    ignore_index=True,
                                    sort=False)

        self.num_styles = num_styles

        if inplace:
            self.exog = self.exog.merge(num_styles, on=style_groups)
        
        return self.exog, style_groups

    def load_sports(self, sports_filepath):
        """Load sports dataframe, assumes monthly grouping and separation
        
        Args:
            sports_filepath (str): filepath for the sports file with games by date
        
        Returns:
            self.sports (pd.df) """
        if sports_filepath.endswith(".xlsx"):
            sports = pd.read_excel(sports_filepath)
        elif sports_filepath.endswith(".csv"):
            sports = pd.read_csv(sports_filepath)

        sports["DateTime"] = pd.to_datetime(sports["days_date"], format="%m/%d/%y")
        sports["Year"] = sports["DateTime"].apply(lambda x: x.year)
        sports["Month"] = sports["DateTime"].apply(lambda x: x.month)
        sports["Week"] = sports["DateTime"].apply(lambda x: x.week)
        sports["Day"] = sports["DateTime"].apply(lambda x: x.day)

        sports.drop(columns=['fiscal_year', "days_date"], inplace=True)

        self.raw_sports = sports

        return self.raw_sports

    def load_beer_styles(self, beer_style_filepath):
        """Load beer styles manually mapped file"""
        if beer_style_filepath.endswith(".xlsx"):
            self.beer_styles = pd.read_excel(beer_style_filepath)
        elif beer_style_filepath.endswith(".csv"):
            self.beer_styles = pd.read_csv(beer_style_filepath)

        return self.beer_styles

    def load_costs(self, cost_filepath=None):
        """Loads national keg cost data

        This method loads the National_KegCosts2019.xlsx file
        and outputs a clean dataframe with unit conversions

        Args:
            cost_filepath (str): path to the cost .xlsx file

        Returns:
            self.costs (pd.df): cleaned cost data
        """
        def unit_size_2_vol(unit_size):
            """
            Convert keg unit size to ounces

            Args:
                unit_size (str): unit size containig "quantity (unit)""

            Returns:
                converted_oz (float): volume in ounces
            """
            converted_oz = 0

            if unit_size != "Missing":
                item_split = unit_size.split()
                quantity = float(item_split[0])
                unit = item_split[1]

                if unit == "gal":
                    converted_oz = quantity * 128
                elif unit == "l":
                    converted_oz = quantity * 33.814
                elif unit == "oz":
                    converted_oz = quantity
                else:
                    converted_oz = np.NaN

            return converted_oz

        cost_converters = {"Segment":str}
        
        if cost_filepath.endswith(".xlsx"):
            costs = pd.read_excel(cost_filepath, converters=cost_converters)
        elif cost_filepath.endswith(".csv"):
            costs = pd.read_csv(cost_filepath, converters=cost_converters)

        costs["KegSize"].fillna('Missing', inplace=True)
        costs["UnitCost"].fillna(costs["UnitCost"].mean(), inplace=True)
        costs["KegSizeOz"] = costs["KegSize"].apply(unit_size_2_vol)
        costs["KegSizeOz"].replace(0, costs["KegSizeOz"].mean(), inplace=True)
        costs["CostPerOz"] = costs["UnitCost"] / costs["KegSizeOz"]
        costs["CostPerOz"].fillna(costs["CostPerOz"].mean(), inplace=True)

        self.costs = costs

        return self.costs

    def load_demo(self, restaurant_filepath=None):
        """Merges restaurant data and census data.

        Args:
            restaurant_filepath (str): path to restaurant info file

        Returns:
            self.demo (pd.df): merged dataframe for restaurant and demo info
        """
        if restaurant_filepath.endswith(".xlsx"):
            restaurant_info = pd.read_excel(restaurant_filepath)
        elif restaurant_filepath.endswith(".csv"):
            restaurant_info = pd.read_csv(restaurant_filepath)

        # Filter necesary rows form restaurant information
        restaurant_cols = ['RestaurantNumber',
                           'LocationName',
                           'City',
                           'State',
                           'Country',
                           'County',
                           'BuildingType',
                           'SeatCountBar',
                           'SeatCountTotal',
                           'Latitude',
                           'Longitude'
                           ]

        restaurant_info = restaurant_info[restaurant_cols]

        # Data profiles for the US census API
        demo_profiles = {'DP05_0037PE': 'White',
                         'DP05_0038PE': 'Black',
                         'DP05_0039PE': 'American_Indian',
                         'DP05_0044PE': 'Asian',
                         'DP05_0052PE': 'Hawaiian',
                         'DP05_0071PE': 'Hispanic',
                         'DP05_0001PE': 'Total_Pop',
                         'DP05_0002PE': 'Male_Pop',
                         'DP05_0003PE': 'Female_Pop',
                         'DP05_0005PE': 'Under_5_years',
                         'DP05_0006PE': '5_to_9_years',
                         'DP05_0007PE': '10_to_14_years',
                         'DP05_0008PE': '15_to_19_years',
                         'DP05_0009PE': '20_to_24_years',
                         'DP05_0010PE': '25_to_34_years',
                         'DP05_0011PE': '35_to_44_years',
                         'DP05_0012PE': '45_to_54_years',
                         'DP05_0013PE': '55_to_59_years',
                         'DP05_0014PE': '60_to_64_years',
                         'DP05_0015PE': '65_to_74_years',
                         'DP05_0016PE': '75_to_84_years',
                         'DP05_0017PE': '85_years_and_over',
                         'DP03_0062E': 'Median_Income'
                         }
        
        demo_vars = list(demo_profiles.keys())
        
        census_info = pd.DataFrame()

        # Iterate through each restaurant and retrieve demo info
        for index, restaurant in restaurant_info.iterrows():
            restaurant_num = restaurant["RestaurantNumber"]
            latitude = restaurant['Latitude']
            longitude = restaurant['Longitude']

            # Using the census geocoding service, retireve the census tract
            geocode_url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
            payload = {"x":str(longitude), 
                       "y":str(latitude),
                       "benchmark":"4",
                       "vintage":"419",
                       "format":"json"}
            result = requests.get(geocode_url, params=payload).json()['result']
            geocode = result['geographies']
    
            tract = geocode['2010 Census Blocks'][0]
            state_fips = tract['STATE']
            county_fips = tract['COUNTY']
            tract_id = tract['TRACT']

            tract_geo_list = [('state', state_fips),
                              ('county', county_fips), 
                              ('tract', tract_id)
                              ]
            
            tract_geo = censusdata.censusgeo(tract_geo_list)
            tract_info = censusdata.download(src='acs5',
                                             year=2018, 
                                             geo=tract_geo, 
                                             var=demo_vars, 
                                             tabletype='profile'
                                             )
            
            tract_info = tract_info.rename(columns=demo_profiles)
            tract_info['RestaurantNumber'] = restaurant_num
            
            census_info = pd.concat([census_info, tract_info],
                                    axis=0, 
                                    ignore_index=True, 
                                    sort=False)

            time.sleep(1) # Limit the rate we query the census

        # Replace census errors or missing values
        census_info = census_info.replace(-888888888, value=np.NaN) 
        census_info = census_info.replace(-666666666, value=np.NaN)
        demo_info = restaurant_info.merge(census_info, 
                                          how='inner', 
                                          on='RestaurantNumber')

        self.demo = demo_info

        return self.demo

    def load_store_info(self, store_info_filepath=None):
        """Cleans the store_list.xlsx file

        Args:
            store_info_file

        Returns:
            self.store_info (pd.df):
        """
        if store_info_filepath.endswith(".xlsx"):
            store_info = pd.read_excel(store_info_filepath)
        elif store_info_filepath.endswith(".csv"):
            store_info = pd.read_csv(store_info_filepath)

        store_info = store_info[store_info["Ownership Type"] == "Corporate"]
        self.store_info = store_info

        return self.store_info

    def group_sales(self, timeframe):
        """Group raw_sales into weekly or monthly groups"""
        if timeframe == "Month":
            groups = ['RestaurantNumber',
                      'Year',
                      'Month',
                      'MonthTime',
                      'SubCategory',
                      'Segment',
                      'VolumeOz',
                      ]

        elif timeframe == "Week":
            groups = ['RestaurantNumber',
                      'Year',
                      'Week',
                      'WeekTime',
                      'SubCategory',
                      'Segment',
                      'VolumeOz',
                      ]       

        aggregates = ['TotalVolumeOz',
                      'DiscPric_Sales',
                      'Price_Sales',
                      'Beer_Orders',
                      ]

        sales = self.raw_sales.groupby(groups, as_index=False).sum()
        self.sales = sales[groups + aggregates]
        self.sales["Price"] = (self.sales["Price_Sales"]
                                / self.sales["Beer_Orders"])

        return self.sales

    def group_sports(self, timeframe):
        """Group sports data to be merged with sales data"""
        time_columns = ["DateTime", "Year", "Month", "Day", "Week"]
        aggregate_mask = ~self.raw_sports.columns.isin(time_columns)
        aggregates = list(self.raw_sports.columns[aggregate_mask])
        groups = ["Year", "Month", "Day"]

        if timeframe == "Month":
            groups = ["Year", "Month"]

        elif timeframe == "Week":
            groups = ["Year", "Week"]

        sports_agg = self.raw_sports.groupby(groups, as_index=False).sum()
        sports_agg = sports_agg[groups + aggregates]

        self.sports = sports_agg

        return self.sports
    
    def group_data(self, timeframe):
        """groups all data by calling the individual helper methods"""
        self.timeframe = timeframe

        self.group_sales(timeframe)
        self.group_sports(timeframe)

        return None

    def get_domestic(self):
        """Returns a list of the top selling domestic beer segments"""
        # Do we need this?
        origin_agg = self.exog.groupby(["SubCategory", "Segment"]).sum()
        domestic = origin_agg.loc["Domestic"].sort_values("Price_Sales", ascending=False)
        domestic_list = list(domestic.index)

        return domestic_list

    def get_base_plan(self, size=10):
        """This method determines the base beer plan. 
        It returns a list of the top selling bers in each style
        And padds out the remaining size of the list with overall best sellers
        
        Args:
            size (int): Size of the base plan
            
        Returns
            self.base_plan (pd.Series): A series of strings of the segment names
        """

        # Find the best sellers for each restaurant and style
        groups = ["Style", "Segment"]
        aggregates = ["TotalVolumeOz", "Price_Sales"]

        segment_data = self.exog.groupby(groups, as_index=False).sum()
        segment_data = segment_data[groups + aggregates]

        style_groups = segment_data.groupby("Style", as_index=False)

        best_styles = {}
        for style, style_data in style_groups:
            best_sellers = style_data.nlargest(n=1, columns="Price_Sales").reset_index()
            best_styles[style] = best_sellers.iloc[0]["Segment"]

        # Find the overall best sellers
        segment_agg = self.exog.groupby("Segment", as_index=False).sum()
        best_sellers = segment_agg.sort_values("Price_Sales", ascending=False).reset_index()
        best_sellers = best_sellers["Segment"]

        base_plan = []
        for style in best_styles:
            if style != 'Other':
                base_plan.append(best_styles[style])

        ranking = 0
        while len(base_plan) < size:
            if best_sellers[ranking] not in base_plan:
                base_plan.append(best_sellers[ranking])
            ranking += 1

        self.base_plan = base_plan

        return self.base_plan

    def separate_exog(self, subcategory=True):
        """Separate the exog df into base plan beers and style groupings"""
        base_mask = self.exog["Segment"].isin(self.base_plan)
        base_exog = self.exog[base_mask]
        style_exog = self.exog[~base_mask]

        if "MonthTime" in self.exog.columns:
            style_groups = ["Style",
                            "RestaurantNumber",
                            "MonthTime",
                            "VolumeOz",
                            "Year"
                            ]

        elif "WeekTime" in self.exog.columns:
            style_groups = ["Style",
                            "RestaurantNumber",
                            "WeekTime",
                            "VolumeOz",
                            "Year"
                            ]
        if subcategory:
            if "MonthTime" in self.exog.columns:
                style_groups = ["Style",
                                "SubCategory",
                                "RestaurantNumber",
                                "MonthTime",
                                "VolumeOz",
                                "Year"
                                ]

            elif "WeekTime" in self.exog.columns:
                style_groups = ["Style",
                                "RestaurantNumber",
                                "SubCategory",
                                "WeekTime",
                                "VolumeOz",
                                "Year"
                                ]

        style_exog_avg = style_exog.groupby(style_groups, as_index=False).mean()

        self.base_exog = base_exog
        self.style_exog = style_exog
        self.style_exog_avg = style_exog_avg
    
        return self.base_exog, self.style_exog, self.style_exog_avg

    def load_exog(self, exog_filepath=None, 
                  demo=True,
                  store_info=True,
                  costs=True,
                  sports=True,
                  beer_styles=True,
                  intra_portfolio=True):
        """Merges sales and demo dataframe, finalize exogenous variables

        This method will take sales and demographic
        data and merge them for finalized exogenous variables.
        Alternatively, it will read pickle filepath.

        Returns:
            self.exog (pd.df): merged dataframe
        """
        if exog_filepath is None:

            if not self.sales.empty:
                print("Loading sales", end="\r")
                self.exog = self.sales.copy()
                #print(self.exog.shape)

            if not self.store_info.empty and store_info:
                print("Loading store info", end="\r")
                self.exog = self.exog.merge(self.store_info, 
                                            on="RestaurantNumber", 
                                            how="left")
                #print(self.exog.shape)

            if not self.demo.empty and demo:
                print("Loading demo", end="\r")
                self.exog = self.exog.merge(self.demo, 
                                            on="RestaurantNumber")
                #print(self.exog.shape)
                
            if not self.sports.empty and sports:
                print("Loading sports", end="\r")
                self.exog = self.exog.merge(self.sports, 
                                            on=["Year", self.timeframe])
                #print(self.exog.shape)
                
            if not self.beer_styles.empty and beer_styles:
                print("Loading beer styles", end="\r")
                self.exog = self.exog.merge(self.beer_styles, 
                                            on="Segment")
                self.get_base_plan()
                print("Loading NumStyles", end="\r")
                self.load_num_styles(inplace=True)
                #print(self.exog.shape)

            if not self.costs.empty and costs:
                print("Loading costs", end="\r")
                self.exog = self.exog.merge(self.costs, 
                                            on="Segment", 
                                            how='left')

                # keg_orders = self.exog["TotalVolumeOz"] / self.exog["KegSizeOz"]
                # keg_orders = np.ceil(keg_orders)
                # self.exog["TotalCost"] =  keg_orders * self.exog["UnitCost"]
                # self.exog["TotalProfit"] = self.exog["Price_Sales"] - self.exog["TotalCost"]
                #print(self.exog.shape)
                
            if not self.beer_styles.empty and intra_portfolio:
                print("Loading intra_portfolio", end="\r")
                self.load_intra(inplace=True)
                #print(self.exog.shape)

        elif exog_filepath is not None:
            self.exog = pd.read_pickle(exog_filepath)
            self.get_base_plan()
            
        self.exog.dropna(inplace=True)
        return self.exog

if __name__ == "__main__":
    pass



