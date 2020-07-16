import pandas as pd
from math import *
import time
import pickle
from pathlib import Path

class Restaurant():

    def __init__(self,DS,RS,restnum, tap_cap = None, name = None, state = None):
        self.DS = DS
        self.RS = RS
        self.restnum = restnum
        self.proxy_dict = None
        self.plan_profits = {}
        self.promo_dict = {}
        self.tap_capacity = int(tap_cap)
        self.name = name
        self.state = state
        # self.demo = None
        # self.optimization = None

    def gen_promo(self):
        """
        Cleans the promo.xlsx filee from the /GT/data/promo directory

        returns:
            dict - {state (str): {month {int}: beer type}}
        """

        # Extra Info
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        listofmonths = {months[n]:n + 1 for n in range(12)}

        d = Path.cwd()
        promos = pd.read_excel([f for f in d.joinpath("data/promo").iterdir() if "xlsx" in f.name][0])
        unique_states = promos["State"].unique()

        promo_dict = {}
        for state in unique_states:
            if state not in ["California - Franchise", "California - NorCal", "Washington D.C."]:
                promo_dict[state] = {}
                for month in range(1,13):
                    promo_dict[state][month] = []
        for index, row in promos.iterrows():
            if row["State"] not in ["California - Franchise", "California - NorCal", "Washington D.C."]:
                promo_dict[row["State"]][listofmonths[row["Month"]]].append(row["Name"])
        self.promo_dict = promo_dict

    def build_proxys(self,num_beers,read_cache=True):
        """
        This method is called to build the various proxy plans which will be
        used to calculate the marginal values for each restaurant/month/segemnt/size 
        combination

        Arguments:
        - self
        - num_beers (int) - number of additional beers you would like to add to base plan

        Results:
        - Restaurant.proxy_dict (dict) - dictionary of proxy plans with structure: 
        {month: {(segment,beer_num): {promotion: proxy plan}}}
        """
        def build_base_proxy(self,month,restnum):
            """
            This helper function is called to build the initial proxy plan 
            containing only the base plan beers for the restaurant/month combination

            Arguments:
            - self
            - month (int) - which month is this proxy plan being created for
            - restnum (int) - which restaurant number is this proxy plan being created for

            Results:
            - proxydf (pd.DataFrame) - dataframe of proxy plan containing only base beers
            """
            df = self.DS.exog
            df = df[df.RestaurantNumber==restnum]
            proxydf = pd.DataFrame()
            # iterate through base plan beers and create dataframe to add to proxy
            for base in self.DS.base_plan:
                for size in self.RS.model_list[base]:
                    promo=0
                    data = df[(df.Segment==base) & (df.VolumeOz==size) & (df.Month==month)]
                    # find price to add to row
                    price = data.mean().Price
                    # get March Madness games
                    # makes sure doesn't overwrite with NaN if data is empty dataframe
                    if not data.empty:
                        march_madness = data.mean().NCAA_BB_PST_games
                    # check if data is an empty dataframe or not
                    if price > 0:
                        style = data.Style.unique()[0]
                        subcat = data.SubCategory.unique()[0]
                    else:
                        try:
                            data = df[(df.Segment==base) & (df.VolumeOz==size)]
                            price = data.mean().Price
                            style = data.Style.unique()[0]
                            subcat = data.SubCategory.unique()[0]
                        except:
                            continue
                               
                    # change price if base plan beer on promotion
                    if ((size==23)&(base in self.promo_dict['Georgia'][month])):
                        promo=1
                        if subcat == 'Domestic':
                            price = 3
                        elif subct == 'Craft':
                            price = 5
                        elif subcat == 'Import':
                            price = 5
                    # append rows to proxy dataframe
                    proxydf = proxydf.append({'Segment':base,'VolumeOz':size,'Style':style,'Price':price,'RestaurantNumber':restnum,'MonthTime':36+month,'Month':month,'NCAA_BB_PST_games':march_madness,'Promo':promo},ignore_index=True)
            return proxydf

        def add_num_styles(proxydf,style,beer_num):
            """
            This helper function is called to add an additional column to the
            proxy plan whose values will be the respective NumStyles value for
            the given proxy plan

            Arguments:
            - proxydf (pd.DataFrame) - completed proxy plan for group iterating over
            - style (string) - style/group for which proxy plan is being built
            (i.e. Lager/Ale/etc.)
            - beer_num (int) - how many additional styles of this beer is being added
            to proxy plan. Used for finding marginal value of adding i'th beer of 
            given style

            Returns:
            - proxydf (pd.DataFrame) - same proxydf inputted but contains additional
            column of NumStyles
            """
            # find number of styles and add to new column "NumStyles"
            a = proxydf.groupby(['Segment','Style']).sum().reset_index().groupby('Style').count()
            # rename column
            a.rename(columns={'Segment':'NumStyles'},inplace=True)
            a = a[['NumStyles']]
            # add extra beer styles to proxy to find marginal value for i'th beer of given style
            a.loc[style] = a.loc[style] + beer_num
            a = a.reset_index()
            
            # merge a onto proxydf by Style so that each row has correct NumStyles value
            proxydf = proxydf.merge(a,on='Style')
            return proxydf
        
        if read_cache:
            filename = 'support/dev/Pickles/proxyDictionaries/'+str(self.restnum)+'_proxy_dict.pickle'
            # try opening pickle instance of proxy plans
            print("Filename : " + filename)
            try:
                with open(filename,'rb') as handle:
                    proxy_dict = pickle.load(handle)
                print('Successfully opened cached proxy_dict.pickle')
                self.proxy_dict = proxy_dict
                return 0
            except:
                print('Could not open cached proxy_dict.pickle')

        # load data
        df = self.DS.exog
        # initialize marginal value dictionary
        proxy_dict = {}
        # get data for given restaurant
        restaurant_data = df[df.RestaurantNumber==self.restnum]
        # get unique segments in dataframe that are not in base plan
        listofsegments = list(set(restaurant_data.Segment.unique()) - set(self.DS.base_plan))
        for month in range(1,13):
            # build stock dataframe with only base plan beers
            base_proxy = build_base_proxy(self,month,self.restnum)
            proxy_dict[month] = {}
            # iterate through segments
            for segment in listofsegments:
                seg_data = df[(df.RestaurantNumber==self.restnum)&(df.Month==month)&(df.Segment==segment)]
                # check if seg_data is empty and redefine if necessary
                if(seg_data.empty):
                    seg_data = df[(df.RestaurantNumber==self.restnum)&(df.Segment==segment)]
                # keep track of progress in your terminal
                print(self.restnum,month,segment)
                for beer_num in range(num_beers):
                    proxy_dict[month][(segment,beer_num+1)] = {}
                    for promo in range(2):
                        # define promotion tag in proxy_dict
                        if promo==0:
                            promo_flag = 'No_Promo'
                        else:
                            promo_flag = 'Promo'
                        # initialize proxy dataframe to contain only base beers
                        proxydf = base_proxy
                        # initialize the number of March Madness games in given month
                        # Taken from base plan because all rows will have same number of games for given month
                        march_madness = proxydf.loc[0,'NCAA_BB_PST_games']
                        for size in seg_data.VolumeOz.unique():
                            size_data = seg_data[seg_data.VolumeOz==size]
                            style = size_data.Style.unique()[0]
                            # only 'TALL' beers go on promotion
                            if ((promo == 1)&(size==23)):
                                # get most common subcategory for group
                                # used for promotional prices
                                subcat = list(size_data.SubCategory.mode())
                                # if Lager, promotion price = 3
                                if 'Domestic' in subcat:
                                    price = 3
                                # if Ale, promotion price = 5
                                elif 'Craft' in subcat:
                                    price = 5
                                # if Import, promotion price = 5
                                elif 'Import' in subcat:
                                    price = 5
                                else:
                                    price = float(size_data.groupby('Segment').mean().reset_index().Price)
                            else:
                                price = float(size_data.groupby('Segment').mean().reset_index().Price)
                            # append row to base proxy plan
                            proxydf = proxydf.append({'Segment':segment,'VolumeOz':size,'Style':style,'Price':price,'RestaurantNumber':self.restnum,'MonthTime':36+month,'Month':month,'NCAA_BB_PST_games':march_madness,'Promo':promo},ignore_index=True)

                        # add NumStyles column to account for i'th beer added to plan
                        new_proxy = add_num_styles(proxydf,segment,beer_num)
                        # store final proxy dataframe in dictionary
                        proxy_dict[month][(segment,beer_num+1)][promo_flag] = new_proxy

        
        # save dictionary to a pickle instance
        filename = 'support/dev/Pickles/proxyDictionaries/'+str(self.restnum)+'_proxy_dict.pickle'
        with open(filename,'wb') as handle:
            pickle.dump(proxy_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)

        self.proxy_dict = proxy_dict

    def get_profits(self,read_cache=True):
        """
        This method is called to calculate the profit of any given proxy plan

        Arguments:
        - None: Restaurant.build_proxys() returns Restaurant.proxy_dict containing
        dictionary of proxy plans

        Returns:
        - Restaurant.plan_profits (dict) - dictionary of proxy profits with structure:
        {month: {(segment,beer_num): {promotion: proxy plan}}}
        """
        def get_proxy_value(self,proxydf):
            """
            This helper function is used to calculate the profit of a single
            proxy plan

            Arguments:
            - self
            - proxydf (pd.DataFrame) - proxy plan for which to find profit

            Returns:
            - profit (float) - profit value of given proxy plan
            """
            # load cost data and manipulate to get groupings
            cost_df = self.DS.costs
            proxydf['PredictedVolume'] = 0
            proxydf['PredictedProfit'] = 0
            # iterate through proxy plan
            for index,row in enumerate(proxydf.iterrows()):
                row = row[1]
                
                # find most recent sales of specific product at location/time to compare against and use best model
                restrow = row.RestaurantNumber
                monthrow = row.Month
                segrow = row.Segment
                sizerow = row.VolumeOz
                
                
                try:
                    compare_row = self.DS.exog[(self.DS.exog.RestaurantNumber==restrow) & (self.DS.exog.Month==monthrow) & (self.DS.exog.Segment==segrow) & (self.DS.exog.VolumeOz==sizerow)]
                    if compare_row.empty:
                        compare_row = self.DS.exog[(self.DS.exog.RestaurantNumber==restrow) & (self.DS.exog.Segment==segrow) & (self.DS.exog.VolumeOz==sizerow)]
                    if compare_row.empty:
                        compare_row = self.DS.exog[(self.DS.exog.RestaurantNumber==restrow) & (self.DS.exog.Segment==segrow)]
                    if compare_row.empty:
                        compare_row = self.DS.exog[(self.DS.exog.Segment==segrow)]                    
                    compare_row = compare_row[compare_row.Year==int(compare_row.Year.max())]
                    compare_row = compare_row.mean()
                    compare_orders = float(compare_row.Beer_Orders)
                    seg_cost = float(compare_row.CostPerOz)
                    

                    # get prediction using the model generated in RegStore.py
                    model = self.RS.model_list[row.Segment][row.VolumeOz]
                    # try block is used in the case that the month specific model was not trained on segment/size comination
                    try:
                        model_prediction = exp(model['General'].get_prediction(row).summary_frame()['obs_ci_lower'])
                    except:
                        model_prediction = 0
                    
                    # get prediction using average values from empirical data
                    try:
                        # first see if product was sold in the given month
                        avg_prediction = self.RS.avg_orders_dict[restrow][segrow][sizerow][monthrow]
                    except:
                        # if not, take average sales among all months
                        try:
                            keys = self.RS.avg_orders_dict[restrow][segrow][sizerow].keys()
                            total = 0
                            for key in keys:
                                total+=self.RS.avg_orders_dict[restrow][segrow][sizerow][key]
                            avg_prediction = total/len(keys)
                        except:
                            avg_prediction = 0
                            for size in self.RS.avg_orders_dict[restrow][segrow].keys():
                                keys = self.RS.avg_orders_dict[restrow][segrow][size].keys()
                                total = 0
                                for key in keys:
                                    total+=self.RS.avg_orders_dict[restrow][segrow][size][key]
                                avg_prediction += total/len(keys)
                    # calculate MAPE for model and average prediction
                    model_mape = abs(compare_orders-model_prediction)/abs(compare_orders)
                    avg_mape = abs(compare_orders-avg_prediction)/abs(compare_orders)
                    # find lowest MAPE and use that value as prediction for given row in proxydf
                    if model_mape <= avg_mape:
                        prediction = model_prediction
                    else:
                        prediction = avg_prediction
                    # calculate predicted revenue, cost, and profit for given row
                    predicted_rev = prediction * float(row.Price)
                    predicted_cost = prediction * float(row.VolumeOz) * seg_cost
                    predicted_profit = predicted_rev - predicted_cost
                    # add predicted profit to proxydf
                    proxydf.loc[index,['PredictedVolume']] = prediction * float(row.VolumeOz)
                    proxydf.loc[index,['PredictedProfit']] = predicted_profit
                
                except:
                    pass
            # find profit for each segment in proxy
            proxydf = proxydf.groupby('Segment').sum().reset_index()
            plan_value = 0
            plan_volume = 0
            for row in proxydf.iterrows():
                if row[1].Segment not in self.DS.base_plan:
                    plan_value += row[1].PredictedProfit
                    plan_volume += row[1].PredictedVolume
            
            return plan_value,plan_volume
        
        # initialize profit dictionary
        profit_dict = {}
        

        if read_cache:
            # try opening pickle instance of profit_dict.pickle
            filename = 'support/dev/Pickles/profitDictionaries/'+str(self.restnum)+'_profit_dict.pickle'
            try:
                with open(filename,'rb') as handle:
                    profit_dict = pickle.load(handle)
                print('Successfully opened cached profit_dict.pickle')
                self.plan_profits = profit_dict
                return 0
            except:
                print('Could not open cached profit_dict.pickle')

            # try opening proxy_dict.pickle
            # saves time having to build all proxy plans again
            filename = 'support/dev/Pickles/proxyDictionaries/'+str(self.restnum)+'_proxy_dict.pickle'
            try:
                with open(filename,'rb') as handle:
                    proxy_dict = pickle.load(handle)
            except:
                print('Could not open cached proxy_dict.pickle')
                proxy_dict = self.proxy_dict
        else:
            proxy_dict = self.proxy_dict


        # iterate through each plan and calculate plan value
        for month in proxy_dict:
            profit_dict[month] = {}
            for segment,beer_num in proxy_dict[month]:
                profit_dict[month][(segment,beer_num)] = {}
                # keep track of progress in your terminal
                print(self.restnum,month,segment,beer_num)
                for promo in proxy_dict[month][(segment,beer_num)]:
                    proxy = proxy_dict[month][(segment,beer_num)][promo]
                    # get proxy plan value
                    proxy_profit,proxy_volume = get_proxy_value(self,proxy)
                    # add proxy plan profit to profit dictionary
                    profit_dict[month][(segment,beer_num)][promo] = (proxy_profit, proxy_volume)

        # write to pickle
        filename = 'support/dev/Pickles/profitDictionaries/'+str(self.restnum)+'_profit_dict.pickle'
        with open(filename,'wb') as handle:
            pickle.dump(profit_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
        
        self.plan_profits = profit_dict

if __name__ == "__main__":
    # import other scripts for testing
    import datastore as ds
    import regstore as rs

    # initialize start time
    start = time.time()



    # import DS object from datastore.py
    DS = ds.DataStore()
    DS.load_exog('month_exog_final.zip')
    DS.get_base_plan()
    DS.separate_exog(subcategory=True)
    # create Segment column in DS.style_exog_avg
    DS.style_exog_avg['Segment'] = DS.style_exog_avg['Style']
    # rename DS.exog to be concatenated DS.base_exog and DS.style_exog_avg
    DS.exog = pd.concat([DS.base_exog,DS.style_exog_avg],sort=False)
    
    ######################################################################
    ######################################################################


    # # import RS object from regstore.py
    RS = rs.RegStore()
    RS.build_model_list(DS)
    # RS.get_valuation(DS)
    # # RS.get_diagnostics()
    # # RS.make_visuals(DS)


    # ######################################################################
    # ######################################################################

    # initialize number of additional beers added to base plan
    num_beers = 8
    # create Restaurant objects for each restaurant in POS data
    listofrestnums = DS.exog.RestaurantNumber.unique()
    for restnum in [85]:
        R = Restaurant(DS,RS,restnum)
        R.gen_promo()
        R.build_proxys(num_beers)
        R.get_profits()
    
    # find runtime
    end = time.time()
    print(str(round(end-start,2)),'seconds')
    # average runtime for entire script: 3 hours