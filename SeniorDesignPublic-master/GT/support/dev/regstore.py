import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from math import *
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class RegStore():

    def __init__(self):
        self.model_list = None
        self.avg_orders_dict = None
        self.diag = None
    
    def build_model_list(self, DS):
        """
        This method is called to calculate the regression models/average orders
        dictionary to be used when calculating profits in MTOP

        Arguments:
        - DS (Object) - DataStore object whose main attributes are .exog, .base_exog,
        .style_exog_avg, & .costs

        Returns:
        - RS.model_list (dict) - dictionary with structure: {segment: {size: regression model}}
        - RS.avg_orders_dict (dict) - dictionary with structure: 
        {RestNum: {segment: {size: {month: avgerage beer orders}}}}
        """

        # load data
        new_data = DS.exog
        # only 2017 and 2018 data used for valuation
        # new_data = new_data[((new_data.Year == 2017)|(new_data.Year==2018))]

        # # create dataframe for base plan
        # base_plan_df = df[df.Segment.isin(self.base_plan)][['Segment','SubCategory','RestaurantNumber',
        # 'MonthTime','Month','Year','VolumeOz','Price','Beer_Orders','NumStyles','Style','MLB_PST_games',
        # 'NCAA_BB_PST_games','NCAA_FB_Reg_games','NCAA_FB_PST_games', 'NFL_Reg_games', 'NFL_PST_games', 'NHL_PST_games']]

        # # create dataframe for grouped beers
        # group_data = df[~df.Segment.isin(self.base_plan)][['Style','SubCategory','RestaurantNumber','MonthTime',
        # 'Month','Year','VolumeOz','Price','Beer_Orders','NumStyles','MLB_PST_games','NCAA_BB_PST_games',
        # 'NCAA_FB_Reg_games','NCAA_FB_PST_games', 'NFL_Reg_games', 'NFL_PST_games', 'NHL_PST_games']]
        # group_data = group_data[group_data.Style!='Unknown']
        # group_data['Segment'] = group_data['Style']
        # group_data = group_data.groupby(['Segment','SubCategory','RestaurantNumber','MonthTime','Month','Year','VolumeOz','Style']).mean().reset_index()
        # # combine two dataframes
        # new_data = pd.concat([group_data,base_plan_df],sort=False)

        # build models and add to nested dictionary with structure of model_dict[segment][size]=model
        listofsegments = new_data.Segment.unique()
        model_dict = {}
        error_dict = {}
        for segment in listofsegments:
            seg_data = new_data[new_data['Segment']==segment]
            model_dict[segment]={}
            for size in seg_data.VolumeOz.unique():
                size_data = seg_data[seg_data['VolumeOz']==size]
                model_dict[segment][size] = {}

                # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
                # url for all model methods and attributes
                model = smf.ols('np.log(Beer_Orders) ~ Price + C(RestaurantNumber) + MonthTime + C(Month) + NumStyles + NCAA_BB_PST_games - 1', data=size_data).fit()
                model_dict[segment][size]['General']=model
                
                # transform fitted values from log transformed to actual values
                predicted_orders = model.fittedvalues.apply(lambda x: exp(x)).sum()

                # convert to predicted number of kegs for segment/size combination
                predicted_kegs = (predicted_orders*size)/(15.5*128)
                actual_values = size_data.Beer_Orders
                total_kegs = (actual_values.sum()*size)/(15.5*128)
                # find MAPE of number of kegs predicted vs actual observed and store in model_dict
                mape = (abs(total_kegs-predicted_kegs)/abs(total_kegs)) * 100

                model_dict[segment][size]['KEGS_MAPE']=mape


        # last case resort - use monthly average of restaurant/segment/size/month combination for prediction
        avg_orders_dict = {}
        for restaurant in new_data.RestaurantNumber.unique():
            avg_orders_dict[restaurant] = {}
            restdata = new_data[new_data.RestaurantNumber==restaurant]
            for segment in restdata.Segment.unique():
                avg_orders_dict[restaurant][segment] = {}
                seg_data = restdata[restdata.Segment==segment]
                for size in seg_data.VolumeOz.unique():
                    avg_orders_dict[restaurant][segment][size] = {}
                    size_data = seg_data[seg_data.VolumeOz==size]
                    for month in size_data.Month.unique():
                        monthdata = size_data[size_data.Month==month]
                        # get average beer orders for restaurant/segment/size/month combination
                        avg = monthdata.mean().Beer_Orders
                        # store avergae value in avg_orders_dict
                        avg_orders_dict[restaurant][segment][size][month] = avg


        self.model_list = model_dict
        self.avg_orders_dict = avg_orders_dict



        # # below block is predicting revenue generated in 2019
        # for restaurant in [650]:
        #     pred_df = pd.DataFrame()
        #     actual_df = pd.DataFrame()
        #     newdata = new_data[(new_data.Year==2019) & (new_data.RestaurantNumber==restaurant)]
        #     for segment in newdata.Segment.unique():
        #         seg_data = newdata[(newdata.Segment==segment)]
        #         for size in seg_data.VolumeOz.unique():
        #             size_data = seg_data[seg_data.VolumeOz==size]
        #             for month in size_data.Month.unique():
        #                 month_data = size_data[size_data.Month==month]
        #                 for subcat in month_data.SubCategory.unique():
        #                     pred_data = month_data[month_data.SubCategory==subcat]
        #                     pred_data.loc[:,'Real_Revenue'] = float(pred_data.Price) * float(pred_data.Beer_Orders)
        #                     actual_df = actual_df.append(pred_data)
        #                     model = model_dict[segment][size]
        #                     # check if MAPE of regression model is below 10%
        #                     # check to see if model was trained on given month
        #                     # does not use model with MAPE=0 due to low sample size
        #                     model_prediction = exp(model['General'].get_prediction(pred_data).summary_frame()['mean'])
        #                     try:
        #                         avg_prediction = avg_orders_dict[restaurant][segment][size][month]
        #                     except:
        #                         keys = avg_orders_dict[restaurant][segment][size].keys()
        #                         total = 0
        #                         for key in keys:
        #                             total+=avg_orders_dict[restaurant][segment][size][key]
        #                         avg_prediction = total/len(keys)
        #                     true_orders = float(pred_data.Beer_Orders)
        #                     model_mape = abs(true_orders-model_prediction)/abs(true_orders)
        #                     avg_mape = abs(true_orders-avg_prediction)/abs(true_orders)

        #                     if model_mape <= avg_mape:
        #                         prediction = model_prediction
        #                     else:
        #                         prediction = avg_prediction
        #                     revenue = prediction*(float(pred_data.Price))
        #                     # print(pred_data)
        #                     # print(revenue)
        #                     pred_df = pred_df.append({'Segment':segment,'VolumeOz':size,'RestaurantNumber':restaurant,'Month':month,'Predicted_Revenue':revenue},ignore_index=True)
        #     pred_df = pred_df[pred_df.Month==12]
        #     rev_df = pred_df.groupby('Month').sum().Predicted_Revenue
        #     actual_df = actual_df.groupby('Month').sum()
        #     rev_df = pd.concat([rev_df,actual_df],axis=1,sort=False)[['Predicted_Revenue','Real_Revenue']]
        #     print(rev_df)









        # # do predictions and find MAPE
        # pred_df = pd.DataFrame()
        # rev_df = pd.DataFrame()
        # vol_df = pd.DataFrame()
        # for segment in listofsegments:
        #     seg_data = new_data[(new_data['Year']==2019) & (new_data['Segment']==segment)]
        #     for size in seg_data.VolumeOz.unique():
        #         size_data = seg_data[seg_data['VolumeOz']==size]
        #         for restnum in size_data.RestaurantNumber.unique():
        #             rest_data = size_data[size_data['RestaurantNumber']==restnum]
        #             total_vol = 0
        #             pred_vol = 0
        #             for monthtime in rest_data.MonthTime.unique():
        #                 revenue = 0
        #                 actual_rev = 0
        #                 predict_data = rest_data[rest_data['MonthTime']==monthtime]
        #                 preds = predict_data[['RestaurantNumber','MonthTime','Price']]
        #                 try:
        #                     pred_model = model_dict[segment][size]
        #                     prediction = float(pred_model.get_prediction(preds).summary_frame(alpha=0.01)['obs_ci_upper'])
        #                     actual_value = float(predict_data['Beer_Orders'])
        #                     MAPE = abs(actual_value - prediction)/abs(actual_value)
        #                     pred_df = pred_df.append({'Segment':segment,'Size':size,'RestaurantNumber':restnum,'MonthTime':monthtime,'Prediction':prediction,'Actual':actual_value,'MAPE':MAPE},ignore_index=True)

        #                     price = float(preds['Price'])
        #                     actual_rev += float(predict_data['Price_Sales'])
        #                     revenue += prediction*price

        #                     total_vol += actual_value*size
        #                     pred_vol += prediction*size
        #                     rev_df = rev_df.append({'Segment':segment,'Size':size,'RestaurantNumber':restnum,'MonthTime':monthtime,'Predicted_Revenue':revenue,'Actual_Revenue':actual_rev},ignore_index=True)
        #                 except:
        #                     continue
                    
        #             # vol_df = vol_df.append({'Segment':segment,'Size':size,'RestaurantNumber':restnum,'MonthTime':monthtime,'Predicted_Volume':pred_vol,'Actual_Volume':total_vol},ignore_index=True)
        # final_errors = pred_df.groupby(['Segment','Size']).mean()['MAPE'].reset_index()

        # vol_df = vol_df.groupby(['RestaurantNumber','Segment']).sum()[['Predicted_Volume','Actual_Volume']]
        # vol_df['Volume_Difference'] = (vol_df['Actual_Volume'] - vol_df['Predicted_Volume'])/1984

        # total_time = time.time() - start_time
        # print('Ran in',round(total_time,2),'seconds')

    def get_valuation(self, DS):
        from pathlib import Path

        # read in promotions file
        # Extra Info
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        listofmonths = {months[n]:n + 1 for n in range(12)}

        d = Path.cwd()
        promos = pd.read_excel([f for f in d.parent.joinpath("GT/data/promo").iterdir() if "xlsx" in f.name][1])
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


        # get 2019 historical revenue for restaurant/month
        df = DS.exog
        df = df[df.Year==2019]
        actual_revenue = df.copy()
        actual_revenue['ActualCost'] = actual_revenue.Beer_Orders * actual_revenue.VolumeOz * actual_revenue.CostPerOz
        actual_revenue['ActualRevenue'] = actual_revenue.Beer_Orders * actual_revenue.Price
        actual_revenue['ActualProfit'] = actual_revenue.ActualRevenue - actual_revenue.ActualCost
        actual_revenue = actual_revenue.groupby(['RestaurantNumber','Month']).sum().reset_index()[['RestaurantNumber','Month','ActualRevenue','ActualCost','ActualProfit']]


        
        # hardcode MarchMadness values for 2019
        marchmadness = {1:0,2:0,3:124,4:11,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}
        # helper function to add NumStyles column
        def add_num_styles(valuation_df):
            """
            This helper function is called to add an additional column to the
            optimal plan whose values will be the respective NumStyles value for
            the given plan

            Arguments:
            - valuation_df (pd.DataFrame) - completed optimal plan for restaurant/month

            Returns:
            - valuation_df (pd.DataFrame) - same optimal plan inputted but contains additional
            column of NumStyles
            """
            # find number of styles and add to new column "NumStyles"
            a = valuation_df.groupby(['Segment','Style']).mean().reset_index().groupby('Style').sum()
            # rename column
            a.rename(columns={'Number':'NumStyles'},inplace=True)
            a = a[['NumStyles']]
            # add extra beer styles to proxy to find marginal value for i'th beer of given style
            a = a.reset_index()
            
            # merge a onto proxydf by Style so that each row has correct NumStyles value
            valuation_df = valuation_df.merge(a,on='Style')
            return valuation_df

        def get_plan_value(self,proxydf,DS):
            """
            This helper function is used to calculate the profit of
            the optimal plan for restaurant/month

            Arguments:
            - proxydf (pd.DataFrame) - optimal plan for which to find profit

            Returns:
            - profit (float) - profit value of given optimal plan
            """
            # load cost data and manipulate to get groupings
            cost_df = DS.costs
            proxydf['PredictedRevenue'] = 0
            proxydf['PredictedCost'] = 0
            # iterate through proxy plan
            for index,row in enumerate(proxydf.iterrows()):
                row = row[1]

                # find most recent sales of specific product at location/time to compare against and use best model
                restrow = row.RestaurantNumber
                monthrow = row.Month
                segrow = row.Segment
                sizerow = row.VolumeOz
                compare_row = DS.exog[(DS.exog.RestaurantNumber==restrow) & (DS.exog.Month==monthrow) & (DS.exog.Segment==segrow) & (DS.exog.VolumeOz==sizerow)]
                if compare_row.empty:
                    compare_row = DS.exog[(DS.exog.RestaurantNumber==restrow) & (DS.exog.Segment==segrow) & (DS.exog.VolumeOz==sizerow)]
                if compare_row.empty:
                    compare_row = DS.exog[(DS.exog.RestaurantNumber==restrow) & (DS.exog.Segment==segrow)]
                if compare_row.empty:
                    compare_row = DS.exog[(DS.exog.Segment==segrow)]
                compare_row = compare_row[compare_row.Year==int(compare_row.Year.max())].mean()
                compare_orders = float(compare_row.Beer_Orders)
                seg_cost = float(compare_row.CostPerOz)
                

                # # Don't need anymore because using exog costs
                # # No need to create cost_df anymore
                # # find cost/oz for given segment
                # try:
                #     seg_cost = float(cost_df[cost_df.Segment==row.Segment].CostPerOz)
                # except:
                #     # default to $0.07/oz
                #     seg_cost = 0.07




                # use try block because some segments and size combination only occured in 2019
                # i.e. model was not trained on it
                try:
                    # get prediction using the model generated in RegStore.py
                    model = self.model_list[row.Segment][row.VolumeOz]
                    # try block is used in the case that the month specific model was not trained on segment/size comination
                    try:
                        model_prediction = exp(model['General'].get_prediction(row).summary_frame()['obs_ci_lower'])
                    except:
                        model_prediction = 0
                    
                    # get prediction using average values from empirical data
                    try:
                        # first see if product was sold in the given month
                        avg_prediction = self.avg_orders_dict[restrow][segrow][sizerow][monthrow]
                    except:
                        # if not, take average sales among all months
                        try:
                            keys = self.avg_orders_dict[restrow][segrow][sizerow].keys()
                            total = 0
                            for key in keys:
                                total+=self.avg_orders_dict[restrow][segrow][sizerow][key]
                            avg_prediction = total/len(keys)
                        # use this block if restnum/segment/size was not trained on
                        except:
                            avg_prediction = 0
                            for size in self.avg_orders_dict[restrow][segrow].keys():
                                keys = self.avg_orders_dict[restrow][segrow][size].keys()
                                total = 0
                                for key in keys:
                                    total+=self.avg_orders_dict[restrow][segrow][size][key]
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
                    # add predicted profit to proxydf
                    proxydf.loc[index,['PredictedRevenue']] = predicted_rev
                    proxydf.loc[index,['PredictedCost']] = predicted_cost
                except:
                    pass
            # find profit for each segment in proxy
            proxydf = proxydf.groupby('Segment').sum().reset_index()
            plan_rev = 0
            plan_cost = 0
            for row in proxydf.iterrows():
                plan_rev += row[1].PredictedRevenue
                plan_cost += row[1].PredictedCost
            plan_value = plan_rev - plan_cost
            return plan_rev, plan_cost, plan_value
        

        predicted_profits = pd.DataFrame()
        listofrestnums = df.RestaurantNumber.unique()
        for restnum in [85,87,99,109,115,246,279,326,332,352,398,481,484,492,567,579,649,650,670,690,693,699]:
            print(restnum)
            # begin reading in optimal plans and find value of each plan
            filename = 'Optimal_plans/'+str(restnum)+'_optimal_plan.pickle'
            with open(filename,'rb') as handle:
                optimal_plan = pickle.load(handle)

            for month in range(1,13):
                mm = marchmadness[month]
                # generate base plan dataframe which will be used for all valuations
                valuation_df = pd.DataFrame()
                for segment in DS.base_plan:
                    segdata = df[(df.Segment==segment)&(df.RestaurantNumber==restnum)&(df.Month==month)]
                    if segdata.empty:
                        segdata = df[(df.Segment==segment)&(df.RestaurantNumber==restnum)]
                    if segdata.empty:
                        segdata = df[(df.Segment==segment)]
                    subcat = segdata.SubCategory.unique()[0]
                    segdata = segdata.groupby(['VolumeOz','Style']).mean().reset_index()
                    style = segdata.Style.unique()[0]
                    for size in segdata.VolumeOz.unique():
                        promo=0
                        sizedata = segdata[segdata.VolumeOz==size]
                        price = float(sizedata.Price)
                        if ((size==23)&(segment in promo_dict['Georgia'][month])):
                            promo=1
                            if subcat == 'Domestic':
                                price = 3
                            elif subcat == 'Craft':
                                price = 5
                            elif subcat == 'Import':
                                price = 5
                        valuation_df = valuation_df.append({'Segment':segment,'VolumeOz':size,'RestaurantNumber':restnum,'Month':month,'MonthTime':month+24,'Price':price,'Style':style,'NCAA_BB_PST_games':mm,'Number':1,'Promo':promo},ignore_index=True)

                # add optimal beer styles to base plan
                plan = optimal_plan[month]
                plan_styles = {}
                for beer,value in plan:
                    style = beer[0]
                    if style in plan_styles.keys():
                        plan_styles[style] += 1
                    else:
                        plan_styles[style] = 1

                for style in plan_styles.keys():
                    # define counter to ensure only 1 of style
                    # is on promotion if applicable
                    count = 0

                    styledata = df[(df.Segment==style)&(df.RestaurantNumber==restnum)&(df.Month==month)]
                    if styledata.empty:
                        styledata = df[(df.Segment==style)&(df.RestaurantNumber==restnum)]
                    if styledata.empty:
                        styledata = df[(df.Segment==style)]
                    if not styledata.empty:
                        subcat = styledata.SubCategory.unique()
                        styledata = styledata.groupby(['VolumeOz','Style']).mean().reset_index()
                        for size in styledata.VolumeOz.unique():
                            sizedata = styledata[styledata.VolumeOz==size]
                            number = plan_styles[style]
                            for num in range(number):
                                promo=0
                                price = float(sizedata.Price)
                                if((size==23)&(style in promo_dict['Georgia'][month])):
                                    if count < 1:
                                        count += 1
                                        promo=1
                                        if 'Domestic' in subcat:
                                            price = 3
                                        elif 'Craft' in subcat:
                                            price = 5
                                        elif 'Import' in subcat:
                                            price = 5
                            
                                valuation_df = valuation_df.append({'Segment':style,'VolumeOz':size,'RestaurantNumber':restnum,'Month':month,'MonthTime':month+24,'Price':price,'Style':style,'NCAA_BB_PST_games':mm,'Number':number,'Promo':promo},ignore_index=True)
                # add NumStyles column to optimal plan
                valuation_df = add_num_styles(valuation_df)
                # get value of optimal plan
                plan_rev, plan_cost, plan_value = get_plan_value(self,valuation_df,DS)

                predicted_profits = predicted_profits.append({'RestaurantNumber':restnum,'Month':month,'PredictedRevenue':plan_rev,'PredictedCost':plan_cost,'PredictedProfit':plan_value},ignore_index=True)
        
        actual_revenue = actual_revenue.merge(predicted_profits,on=['RestaurantNumber','Month']).groupby('RestaurantNumber').sum().reset_index()
        # actual_revenue.drop('Month')
        actual_revenue['GrossIncrease'] = actual_revenue.PredictedProfit - actual_revenue.ActualProfit
        actual_revenue['PercentIncrease'] = (actual_revenue.GrossIncrease/actual_revenue.ActualProfit)*100
        print(actual_revenue)

        actual_revenue.to_csv('Final_Valuation.csv')
        
    def get_diagnostics(self):
        """
        Run this method to get the diagnostic results for all the models

        Arguments:
        - self

        Returns:
        - self.diagnostics (pd.DataFrame) - dataframe of diagnostic attributes for each model
        """

        # # Residuals vs. Fitted values csvs
        # for segment,size in [('Bud Light',16),('Bud Light',23),('IPA',16)]:
        #     residdf = pd.DataFrame()
        #     fitdf = pd.DataFrame()
        #     filename = segment+str(size)+'.csv'
        #     segdf = pd.DataFrame()
        #     model = self.model_list[segment][size]['General']
        #     model_resid = model.resid
        #     model_fit = model.fittedvalues
        #     for value in model_resid:
        #         residdf = residdf.append({'Residual':value},ignore_index=True)
        #     for value in model_fit:
        #         fitdf = fitdf.append({'Fitted':value},ignore_index=True)
        #     residdf['Fitted'] = fitdf
        #     residdf['Segment'] = segment
        #     residdf['Size'] = size
        #     residdf.to_csv(filename)




        # # Model Validation
        # f_t_df = pd.DataFrame()
        # for segment in self.model_list:
        #     for size in self.model_list[segment]:
        #         model = self.model_list[segment][size]
        #         tval = model['General'].pvalues
        #         r2 = model['General'].rsquared_adj
        #         fpval = model['General'].f_pvalue
        #         tval = tval.append(pd.Series({'R2':r2,'F_pval':fpval,'Segment':segment,'Size':size}))
        #         # mse = model['General'].mse_total
        #         # ssr = model['General'].ssr
        #         # kegs_mape = round(model['KEGS_MAPE'],2)
        #         f_t_df = f_t_df.append(tval,ignore_index=True)
        # f_t_df.to_csv('Model_Validation.csv')

        


        # Visualizations
        for segment,size in [('Bud Light',16),('Bud Light',23),('IPA',16)]:
            figure_title = segment+str(size)+'QQ_Plot'
            model = self.model_list[segment][size]['General']
            model_resid = model.resid
            fig = sm.qqplot(model_resid,fit=True,line='45')
            fig.suptitle(figure_title)
            plt.savefig(figure_title)
            plt.show()
        




    def make_visuals(self, DS):
        data = DS.exog

        mapedf = pd.DataFrame()
        for segment in self.model_list:
            for size in self.model_list[segment]:
                mape = self.model_list[segment][size]['KEGS_MAPE']
                nobs = self.model_list[segment][size]['General'].nobs
                mapedf = mapedf.append({'Segment':segment,'Size':size,'MAPE':mape,'NObs':nobs},ignore_index=True)
        
        sns.relplot(x='NObs',y='MAPE',data=mapedf,hue='Segment')
        plt.show()
        
        # data = DS.exog

        # for restaurant in data.RestaurantNumber.unique()[0:2]:
        #     plotdata = data[(data.RestaurantNumber==restaurant)&(~data.Segment.isin(DS.base_plan)) & (data.Segment=='Lager')]
            
        #     plotdata['Prediction'] = 0 
        #     for index,row in enumerate(plotdata):
        #         model_prediction = exp(model['General'].get_prediction(row).summary_frame()['mean'])
        #         plotdata.loc[index,'Prediction'] = model_prediction
            
            
        #     sns.relplot(x='MonthTime',y='Beer_Orders',hue='SubCategory',kind='line',col='Segment',data=plotdata)
        #     plt.show()



if __name__ == '__main__':
    pass