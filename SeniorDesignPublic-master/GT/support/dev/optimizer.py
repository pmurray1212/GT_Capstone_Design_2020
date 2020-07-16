import pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pandas as pd
from pyomo.core import Var
from timeit import default_timer as timer
import pprint

import random

#temporty import
from pathlib import Path
import csv

class Optimizer():

    def __init__(self, id, tap_capacity, marginal_values, beer_map = {}, base_plan = [], promo_beers = []):
        #super.__init__(id, tap_capacity)
        self.id = id
        self.model = pyo.ConcreteModel()
        self.tap_capacity = tap_capacity
        
        """
        BASE PLAN
        """ 
        self.base_plan = base_plan

        """
        LIST OF BEER STYLES
        """
        styles = []
        for beer,number in marginal_values.keys():
            if beer in styles:
                pass
            else:
                styles.append(beer)
        self.beer_styles = styles

        """
        PROMOTIONAL DATA
        """
        self.bur_promo_input = promo_beers
        
        """
        MARGINAL VALUE DATA
        """
        self.marginal_values = marginal_values

        """ 
        TEMPORARY BEER MAPPING
        """
        self.beer_map_dict = beer_map

        """
        PROMOTIONAL DICTIONARY
        """
        self.beer_promo_dict = {}
        for beer in self.bur_promo_input:
            if beer not in self.base_plan:
                if beer in self.beer_styles:
                    if beer not in self.beer_promo_dict:
                        self.beer_promo_dict[beer] = 1
                    else:
                        self.beer_promo_dict[beer] = self.beer_promo_dict[beer] + 1
                else:
                    #print(beer + " is not in beerstyles")
                    if beer in self.beer_map_dict:
                        if beer not in self.beer_promo_dict:
                            self.beer_promo_dict[self.beer_map_dict[beer]] = 1
                        else:
                            self.beer_promo_dict[self.beer_map_dict[beer]] = self.beer_promo_dict[self.beer_map_dict[beer]] + 1
        
        self.beer_promo_dict2 = self.beer_promo_dict.copy()

        self.mandate_list = None
        self.mandate_list_volume = {}
        self.declare_parameters()
        self.declare_vars()
        self.declare_obj()
        self.declare_constraints()
        self.results = self.solve()
        self.gen_results_list()

    def declare_parameters(self):
        # style - set of all beer styles
        self.model.styles = pyo.Set(initialize = self.beer_styles)

        self.multiplier = 8

        # num - range for each style
        self.model.num_range = pyo.RangeSet(1,self.multiplier)

        # B_{i, n}
        # Marginal change in value for style i tap n
        self.model.marginal_profit_set = pyo.Set(dimen=2, initialize = {(style, num) for style in self.model.styles for num in self.model.num_range})
        def marginal_profit_init(model, style, num):
            if style in self.beer_promo_dict:
                if self.beer_promo_dict[style] != 0:
                    self.beer_promo_dict[style] = self.beer_promo_dict[style] - 1
                    self.mandate_list_volume[(style,num)] = self.marginal_values[(style,num)]["Promo"][1]
                    return self.marginal_values[(style,num)]["Promo"][0]
                else:
                    self.mandate_list_volume[(style,num)] = self.marginal_values[(style,num)]["No_Promo"][1]
                    return self.marginal_values[(style,num)]["No_Promo"][0]
            else:
                self.mandate_list_volume[(style,num)] = self.marginal_values[(style,num)]["No_Promo"][1]
                return self.marginal_values[(style,num)]["No_Promo"][0]
        self.model.marginal_profit = pyo.Param(self.model.marginal_profit_set, initialize = marginal_profit_init)
        #self.model.marginal_profit.pprint()

    def declare_vars(self):
        # x_{i,n} - binary decision variable
        # 1 if style i is on tap n, 0 otw
        self.model.x = pyo.Var(self.model.marginal_profit_set, within=pyo.Binary, initialize = 0)

    def declare_obj(self):
        # Objective
        def objective_rule(model):
            single = sum(self.model.marginal_profit[(style,num)] * self.model.x[(style,num)] for style in self.model.styles for num in self.model.num_range)
            return single
        self.model.Obj = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def declare_constraints(self):

        self.model.cuts = pyo.ConstraintList()

        self.model.cuts.add(sum(self.model.x[(i,n)] for i in self.model.styles for n in self.model.num_range) == self.tap_capacity - len(self.base_plan))
        
        for i in self.model.styles:
            for n in range(1, self.multiplier):
                self.model.cuts.add(self.model.x[(i,n)] >= self.model.x[(i,n+1)])

        for beer in self.beer_promo_dict2:
            for num in range(self.beer_promo_dict2[beer]):
                self.model.cuts.add(self.model.x[(beer, num + 1)] == 1)

        # Force one of each style
        for i in self.model.styles:
            self.model.cuts.add(self.model.x[(i,1)] == 1)
        """
        print("\nBASE PLAN")
        print(self.base_plan)
        print("\nPREV PROMO DICT")
        print(self.bur_promo_input)
        print("\nNEW PROMO DICT")
        print(beer_promo_dict)
        print("\nCONSTRAINTS")
        self.model.cuts.pprint()
        """

    def solve(self):
        print("Solving")
        solver_start = timer()
        opt = pyo.SolverFactory('cbc')  # Select solver
        solver_manager = pyo.SolverManagerFactory('neos')
        results = solver_manager.solve(self.model, opt=opt)
        solver_end = timer()

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            print("This is feasible and optimal")
            print(solver_end - solver_start)
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print ("Do something about it? or exit?")
        else:
            # something else is wrong
            #print (str(results.solver))
            print("store " + str(self.id) + " solved")

        #print(self.bur_promo)

        #results.write(num=1)

        #self.gen_results_list

        #pprint.pprint(var_values)
        #pprint.pprint(var_coefs)
        
        return results
    
    def gen_results_list(self):
        var_values = {}
        var_coefs = {}

        for v in self.model.component_objects(Var, active=True):
            varobject = getattr(self.model, str(v))
            beer_values = []
            coefs_values = []
            for index in varobject:
                if varobject[index].value != 0:
                    #print ("   ",index, varobject[index].value)
                    beer_values.append(index)
                    coefs_values.append((index, self.model.marginal_profit[index], self.mandate_list_volume[index]))
            var_values[str(v)] = beer_values
            var_coefs[str(v)] = coefs_values
        
        """
        coefs = []
        for v in ["gamma", "x"]:
            for item in var_coefs[v]:
                coefs.append(item[-1])
        """
        #print(self.bur_promo_input)
        self.mandate_list = {beer[0]:(beer[1],beer[2]) for beer in var_coefs["x"]}
        

if __name__ == "__main__":
    o = Optimizer("25", 33, None)