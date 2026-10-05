# 
# !pip install pymoo
# !pip install osmnx


import random
import datetime
import time

import multiprocessing
import numpy as np
import math

import argparse

import copy
import geopandas as gdp
import osmnx as ox
import networkx as nx
import pandas as pd
from pathlib import Path

# import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

# flatten(get_params(moead))

# You only can minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Parallel Slave-Master
try:
    # pymoo versions used by the original project
    from pymoo.core.problem import StarmapParallelization
except ImportError:
    # pymoo >= 0.6.2
    from pymoo.parallelization.starmap import StarmapParallelization

# Biased initialization
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

MALAGA_SCENARIO = "Malaga-Subway"
MELILLA_SCENARIO = "Melilla"
MALAGA_INPUT_FILE = "map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY-wBUS-wMetro"
MELILLA_INPUT_FILE = "map-Melilla-all--scooter-walking-bus"

# CONSTANT_COST = 0.00048 # Millions of euros per metre

# MIN_COST = 0
# MAX_COST = 1171337.95399999 #* CONSTANT_COST
# MIN_TIME = 1393.04855117
# MAX_TIME = 1745.86999262

# EVO* CONSTANTS
# MIN_COST = 0
# MAX_COST = 125704.01099999978 # * CONSTANT_COST ( aprox 60,337,925.27 €)
# MIN_TIME = 1393.048551165003
# MAX_TIME = 1745.8699926158945

# New Map CONSTANTS
MIN_COST_MALAGA = 0
MAX_COST_MALAGA = 125869.75899999992 # * CONSTANT_COST ( aprox 60,337,925.27 €)
MIN_TIME_MALAGA = 1392.8303198078484
MAX_TIME_MALAGA = 1715.5414956736442

MIN_RATIO_MALAGA = 0 #0.017028428700404805
MAX_RATIO_MALAGA = 1 #0.05255499331895058 
# MIN_RATIO = 0
# MAX_RATIO = 1

MIN_RATIO_MELILLA = 0 #0.00305848482852833
MAX_RATIO_MELILLA = 1 #0.243129508086676

MIN_COST_MELILLA = 0 # 0.00305848482852833 Antes del 14 de oct
MAX_COST_MELILLA = 43842.381

MAX_TIME_MELILLA = 1054.43302841235
MIN_TIME_MELILLA = 832.074560896067



## NUMBER OF OBJECTIVES
NUMBER_OBJECTIVES = 3 # 

# # I/O Configuration
import sys
IN_COLAB = 'google.colab' in sys.modules

base_path = ''
# if IN_COLAB:
#     from google.colab import drive
#     drive.mount('/content/gdrive')
#     base_path = '/content/gdrive/Shareddrives/happy_mob'
#     get_ipython().system('ls /content/gdrive/Shareddrives/happy_mob/')
# else:


TIME_WEIGHT = "time_experiment"
# # Parameter of Algorithms



# # Fitness evaluation

def get_distances(map_graph, sections, schools, weight_metric="time_w_c_m"):
    """
        Get Distances in a list
    """    
    distances_weight = list()
    
    for orig_node in sections:
        for dest_node  in schools:
            # print('\tSchools to process: {}'.format(schools_to_process))
            distance, _ = get_route_and_distance(map_graph, orig_node, dest_node, weight_metric)
            distances_weight.append(distance)
            
    return distances_weight



def pair_to_list_tuples(pair_list):
    list_tuples = []
    sections = set()
    schools = set()

    for i in pair_list.itertuples():
        school = getattr(i,"schools")
        section = getattr(i,"sections")

        schools.add(school)
        sections.add(section)

        list_tuples.append( (school, section) )

    return list_tuples



# ## Some useful functions

# Function defined to get the route and the distance between two points using Dijkstra
def get_route_and_distance(map_graph, origin_id, dest_id, weight_metric):
    '''
        @param map_graph : Graph
        @param origin_id: source, Node id of origin
        @param dest_id: target, Node id of destination
        @param weight_metric : String, Function or None, that represent the objective to minimize
            Function should be (\\u,v,data -> number)

        @return distance: distance [in weight] of the route
        @return path : list of nodes from source to target
    '''

    return  nx.bidirectional_dijkstra(map_graph, origin_id, dest_id, weight_metric)

def get_distances_pair(map_graph, pair_list, weight_metric=TIME_WEIGHT, verbose=False):
    """
        Function return the distances of serie of O-D points in a list of tuples

        If verbose is True:
          @return Dataframe of the distances [in weight] of each route
          @return Dataframe of the osmid edges that makes each route
          @return String of pair (origin,destination) that causes errors
          @return Number of routes processed (orign x destination)
          @return Dataframe of the osmid edges that makes each route
          @return Dataframe of the type of edges (follow the prioritize list) taken in each routes
        else:
           @return List of the distances [in weight] for each pair
           @return List of metric (change_mode_e-scooter-walking/number_edges-e-scooter-walking) for each pair

        @param map_graph  : Graph to search the routes
        @param pair_list  : datframe, where schools are the node origns and sections are the destination node
        @param weight_metric   : String or Function (u,v,data)-> number to represent the weight
        @param verbose : Boolean for more information, default False
    """


    # if(verbose):
        # routes_to_proccess = len(pair_list)
        # districts_to_process = len(sections)
        # print('Routes to process: {}'.format(routes_to_proccess))
        # route_type_dict = dict() - LEGACY
        # route_osmid_edges_dict = dict()  - LEGACY
        # times_routes_dict  = dict() - LEGACY
        # route_type_dict['dest'] = "time" - LEGACY

        # Add the destination
        # route_osmid_edges_dict['dest'] = "time"  - LEGACY
        # times_routes_dict['dest'] = "time" - LEGACY
        # total_orign_target_viewed = 0 - LEGACY
        # error = "orig_node,dest_node\n"
        # distances_dict = dict() - LEGACY
        # distances_dict['dest']= "time" - LEGACY



    distances_weight = list() # Save the total weight of the shortest route
    
    list_ratio_change_mode_number_edges = list()  # Save the list of change_mode_active/# edges #NEEDED
    list_edges_that_changed = list() # Save the list of all edges of e-scooter to obtain the cost #NEEDED
    

    for i in pair_list:
        # if verbose:
        #     # distances_weight = list() # Save the total weight of the shortest route - LEGACAY
        #     # LEGACY
        #     # route_type_list = list() # Save how the routes are compound
        #     # times_routes = list() # Save the tuple of how many times walk, scooter, and total
        #     # route_osmid_edges_list = list()

        #     route_osmid_edges = ""

        orig_node = i[0]
        dest_node = i[1]
        try:
            # Map should be a directed graph with weight in non-negative integer
            distance, route_of_nodes = get_route_and_distance(map_graph,orig_node,dest_node, weight_metric) # get distance and routes
            route_edges_attr = ox.utils_graph.route_to_gdf(map_graph, route_of_nodes,weight=weight_metric) # list of Edges attributes of route

            # Save the time
            distances_weight.append(distance)




            #######################################################
            ## CALCULATE THE CHANGE e-scoooter-walking / EDGES   ##
            #  AND GET THE U,V,K of those edges that changed     ##
            #######################################################
            # Calculated the list of changes/#edges_walk+scooter
            is_before_escooter = False
            is_before_walk = False
            change_mode_walk_escooter = 0 # If equal to 0, then only was bus or subway or (only one way)
            times_scooter = 0
            times_walk = 0
            # if(verbose):
            #     route_osmid_edges = '-' # OSMID Route

            for edge in route_edges_attr.itertuples():

                if (verbose):
                    route_osmid_edges = route_osmid_edges  + '{}-'.format(getattr(edge, 'osmid'))

                edge_walk = getattr(edge, 'edge_walk', None)
                edge_scooter  = getattr(edge, 'edge_scooter', None)
                edge_subway = getattr(edge, 'edge_subway', None)
                edge_bus = getattr(edge, 'edge_bus', None)
                edge_changed = getattr(edge, 'edge_changed', None)



                if bool(edge_subway) or bool(edge_bus):
                    is_before_escooter = False
                    is_before_walk = False
                    # if(verbose):
                    #     print(f"edge subwway={bool(edge_subway)} and edge_bus={bool(edge_bus)}")

                elif bool(edge_scooter) or bool(edge_changed):
                    times_scooter += 1
                    is_before_escooter = True # Now, I am using the e-scooter
                    if is_before_walk:
                        # if(verbose):
                        #     print("\t********before was walking?, then change")

                        change_mode_walk_escooter +=1
                    is_before_walk = False
                    # if(verbose):
                    #     print(f"e-scooter")

                elif bool(edge_walk):
                    times_walk += 1
                    is_before_walk = True # Now, I am walking
                    if is_before_escooter:

                        # if(verbose):
                        #     print("\t********before was e-scooter?, then change")
                        change_mode_walk_escooter += 1
                    is_before_escooter = False

                    # if(verbose):
                    #     print(f"walk")
                else:
                    # if(verbose):
                    #     print("********************")
                    #     print("ERROR")
                    #     print(f"edge_walk = {edge_walk}; edge_scooter  = {edge_scooter}")
                    #     print(f"edge_subway = {edge_subway}; edge_bus = {edge_bus}")
                    #     print(f"edge_drive = {edge_changed}")
                    #     print(edge)
                    #     print("**********************")
                    pass

                # End for each edge of the route attr
                # if(verbose):
                #     print("-###################################-")
                #     print("-###################################-")
                #     print(f"Change mode e-scooter/walking {change_mode_walk_escooter}")
                #     print(f"Number of edges e-scooter or walking {times_scooter + times_walk}")
                #     print(route_osmid_edges)

                ratio_change_mode_edges_e_w = change_mode_walk_escooter/(times_scooter+times_walk)

                list_ratio_change_mode_number_edges.append(ratio_change_mode_edges_e_w)
                # End the calcutaion of the ratio (inside for)
                # ----------------------------------------------- #
                # (Still inside for)
                # Start to obtain the u,v,k of those edges that changed

                ss_boolean = getattr(edge,"search_space",None)
                edge_change = getattr(edge,"edge_changed",None)

                if( bool(ss_boolean) and bool(edge_change)):
                    u_escooter,v_escooter,k_escooter = getattr(edge,"Index")
                    u_v_k_escooter = (u_escooter,v_escooter,k_escooter)
                    if not(u_v_k_escooter in list_edges_that_changed): # Saving time
                        list_edges_that_changed.append(u_v_k_escooter)

                # LEGACY
                # if(verbose):
                #     len_all_edges_route += len(route_edges_attr)


            #######################################################
            ##      END CALCULATING RATIO AND LIST_EDGES         ##
            #######################################################


        except:
            # if(verbose):
            #     error_message = f'{orig_node},{dest_node}\n'
            #     error += error_message
            #     print(error_message)
            pass


        ###############################
        ## This should be outside the try/except because
        ## All (array=DF) must be of the same length
        ###############################





        # LEGACY CODE
        # if(verbose):
        #     route_type_list.append(route_type)
        #     # length_distances.append(length_distance)
        #     # try:
        #         # times_routes.append( (times_walk, times_scooter, route_size))
        #         # except:
        #         # print("There is a problem with route_size because the route doesn't exist")
        #     route_osmid_edges_list.append(route_osmid_edges)

        #     distances_dict[i] = distances_weight
        #     route_type_dict[i] = route_type_list
        #     # length_distances_dict[i] = length_distances
        #     # times_routes_dict[i] = times_routes
        #     route_osmid_edges_dict[i] = route_osmid_edges_list

        #     routes_to_proccess -= 1
        #     # print('Routes to process: {}'.format(routes_to_proccess))

    # break


    # End for each orign nod in section

    # LEGACY CODE
    # if(verbose):
    #       return pd.DataFrame(distances_dict), pd.DataFrame(route_osmid_edges_dict), \
    #                     error, total_orign_target_viewed, \
    #                     pd.DataFrame(route_type_dict)
    # else:
    #        return distances_weight, list_fc


    return distances_weight, list_ratio_change_mode_number_edges, list_edges_that_changed

def get_distances_pair_mean(map_graph, pair_list, weight_metric=TIME_WEIGHT, verbose=False):
    """        Function return the mean of distances of serie of O-D points in a list of tuples
    """
    #list_distances = get_distances(map_graph,schools=schools, sections=sections, weight_metric=TIME_WEIGHT)
    list_distances, list_ratio_change_mode_number_edges, list_edges_that_changed  = get_distances_pair(map_graph, pair_list, weight_metric=TIME_WEIGHT, verbose=False)
    # print(f"list_distances={list_distances}")
    # print(f"list_ratio_change_mode_number_edges={list_ratio_change_mode_number_edges}")
    # print(f"list_edges_that_changed={list_edges_that_changed}")
    to_return_mean_distances = np.mean(list_distances)
    to_return_mean_ratio = np.mean(list_ratio_change_mode_number_edges)
    
    return to_return_mean_distances, to_return_mean_ratio, list_edges_that_changed


# # Evaluation function (Fitness)
def get_cost(individual,list_edges_that_changed,evostarcost):
    cost = 0
    if not(evostarcost):
        # indv_intersect = [0 for i in range(LEN_SEARCH_SPACE)]
        unique = set(list_edges_that_changed)
        for u,v,k in unique:
            pos = dict_of_search_space_uvk_to_pos[(u,v,k)]
            if(individual[pos]):
                cost_temp = dict_of_search_space_cost[pos]
                if(cost_temp <=0):
                    print("There is something bad here!!!")
                cost += cost_temp
                
    else:
        for i in range(0,len(individual)):
            if(individual[i]):
                cost_temp = dict_of_search_space_cost[i]
                if(cost_temp <=0):
                    print("There is something bad here!!!")
                cost += cost_temp
    
    
    return cost


def evaluation(individual, list_pair_od, extremes_min_max):
    #start_time_eva = time.time() 
    
    # To Normalize
    MIN_TIME = extremes_min_max["MIN_TIME"]
    MAX_TIME = extremes_min_max["MAX_TIME"]
    
    MIN_COST = extremes_min_max["MIN_COST"]
    MAX_COST = extremes_min_max["MAX_COST"]
    
    MIN_RATIO = extremes_min_max["MIN_RATIO"]
    MAX_RATIO = extremes_min_max["MAX_RATIO"]
    
    
    
    bounded = copy.deepcopy(G)
    # #Copio cada vez un grafo distinto
    # # len(individual) == LEN_SEARCH_SPACE
    for i in range(0,len(individual)):
        if(individual[i]):
            u,v,k = dict_of_search_space_u_v[i]
            data = bounded.get_edge_data(u, v, key=k)
            if(data == None):
                print("We have a situation here")
            else:
                data["time_w_c"] = data["time_experiment_new"]
                data[TIME_WEIGHT] = data["time_experiment_new"]
                data["edge_changed"] = True 

    #print("\t\t\tObteniendoDistancias")
    objective1, objective3, list_edges_that_changed = get_distances_pair_mean(map_graph=bounded, pair_list=list_pair_od, weight_metric=TIME_WEIGHT)

    objective1 = (objective1-MIN_TIME)/(MAX_TIME-MIN_TIME)
   
    
    objective2 = get_cost(individual,list_edges_that_changed,evostarcost=False) #* CONSTANT_COST
    objective2 = ((objective2-MIN_COST)/(MAX_COST-MIN_COST))
 
    objective3 = (objective3-MIN_RATIO)/ (MAX_RATIO - MIN_RATIO)
 
    # print(f"{np.sum(individual)}->({objective1}, {objective2})")
    #print('\t\t\tOne evaluation in POB {} NGEN{} in seconds {} CXPB {} MUFLIP {} CPUS {}'.format(MU,NGEN,float(end_time_eva-start_time_eva),CXPB, MUFLIP,cpus))
    return objective1, objective2, objective3


#################################################
##  
##  PROBLEM CREATION PYMOO STYLE
##
#################################################

class VectorOneMinimizeParallel(ElementwiseProblem):

    # basic setups
    # # n_var -> number of variables
    # # n_obj -> number of objectives
    # # xl -> lower boundaries of variables
    # # xu -> upper boundaries of variables
    def __init__(self, size_vector, list_pair_od, scenario, **kwargs):
        self.extremes_min_max = {}
        
        if (scenario == "MALAGA"):
            self.min_cost = MIN_COST_MALAGA
            self.max_cost = MAX_COST_MALAGA
            
            self.min_ratio = MIN_RATIO_MALAGA
            self.max_ratio = MAX_RATIO_MALAGA
            
            self.min_time = MIN_TIME_MALAGA
            self.max_time = MAX_TIME_MALAGA
        elif (scenario == "MELILLA"):
            self.min_cost = MIN_COST_MELILLA
            self.max_cost = MAX_COST_MELILLA
            
            self.min_ratio = MIN_RATIO_MELILLA
            self.max_ratio = MAX_RATIO_MELILLA
            
            self.min_time = MIN_TIME_MELILLA
            self.max_time = MAX_TIME_MELILLA
        else:
            sys.exit("No Scenario valid/ in VectorOneMinimizeParallel Inititialization")
            
        print(f"COST->MIN:{self.min_cost} MAX {self.max_cost}")
        print(f"RATIO->MIN:{self.min_ratio} MAX {self.max_ratio}")
        print(f"TIME->MIN:{self.min_time} MAX {self.max_time}")
    
        self.extremes_min_max["MIN_COST"] = self.min_cost
        self.extremes_min_max["MAX_COST"] = self.max_cost
        
        self.extremes_min_max["MIN_RATIO"] = self.min_ratio
        self.extremes_min_max["MAX_RATIO"] = self.max_ratio
        
        self.extremes_min_max["MIN_TIME"] = self.min_time
        self.extremes_min_max["MAX_TIME"] = self.max_time
        
        self.list_pair_od = list_pair_od
        super().__init__(n_var=size_vector,
                         n_obj=3,
                         xl=0,
                         xu=1, vtype=int, **kwargs)

    #  x, out are needed, it's basic
    #  x -> solutions, for this kind of Problem, the x is a (population, n_var) matrix. 
    #  out -> the corresponding results, including function values, constrain values and so on. 
    #        we only fill what we need. 
    def _evaluate(self, x, out, *args, **kwargs):
        # # out["F"] -> Fitness
        # # out["G"] -> Constrains values
        # out["F"] = (np.random.random(),np.random.random()) 
        
        # print("\tI am doing an evaluation\n",flush=True)
        
        objective1, objective2, objective3 = evaluation(x.astype(int), list_pair_od=self.list_pair_od, extremes_min_max=self.extremes_min_max)
        out["F"] = tuple((objective1, objective2, objective3))




class FileOutput(Output):
    
    def __init__(self,headers, filepath=None, show_ind=False):
        """_summary_

        Args:
            header (List String): List of columns name. If show_ind is True, 
                last_column is the name of the indivuals
            filepath (Path, optional): Filepath to save the fitness values. Defaults to None.
            show_ind (bool, optional): Enable the display of indivudual representation. Defaults to False.
        """
        super().__init__()
        self.filepath = filepath
        self.headers = headers

        # print(self.headers)
        if(filepath is not None):
            previous_df = pd.DataFrame({}, columns=headers)
            previous_df.to_csv(self.filepath, index=False)
            # print(f"Filepath_Output = {self.filepath}")
    
        self.show_ind = show_ind
        self.x_mean = Column("x_mean", width=21)
        self.x_std = Column("x_std",  width=21)
        self.x_min = Column("x_min", width=21)
        self.x_max = Column("x_max", width=21)
        # self.columns = [self.n_gen] #[self.n_gen, self.n_eval] # Dont care about n_eval
        
        if(self.show_ind):
            self.x_ind = Column(headers[-1], width=13,truncate=False)
            self.columns += [self.x_mean, self.x_std, self.x_min, self.x_max, self.x_ind]
        else:
            self.columns += [self.x_mean, self.x_std, self.x_min, self.x_max]

    def update(self, algorithm):
        super().update(algorithm)

        print(f"Gen {algorithm.n_gen}: pop size = {len(algorithm.pop)}")
        print(f"F shape: {algorithm.pop.get('F').shape}")

        if(self.show_ind):
            columns_fitness = self.headers[1:-1]
        else:
            columns_fitness = self.headers[1:]
        

        fitness_df = pd.DataFrame(algorithm.pop.get("F"), columns=columns_fitness)
        
        fitness1 = fitness_df["fitness1"].values
        fitness2 = fitness_df["fitness2"].values
        fitness3 = fitness_df["fitness3"].values
        self.x_mean.set( 
            tuple((round(np.mean(fitness1),3),
                  round(np.mean(fitness2),3),
                  round(np.mean(fitness3),3)))
            )
        self.x_min.set(
                       tuple((round(np.min(fitness1),3),
                            round(np.min(fitness2),3),
                            round(np.min(fitness3),3)))
            
                       )
        self.x_max.set(
                       tuple((round(np.max(fitness1),3),
                            round(np.max(fitness2),3),
                            round(np.max(fitness3),3)))
            
                       )
        self.x_std.set(
                       tuple((round(np.std(fitness1),3),
                            round(np.std(fitness2),3),
                            round(np.std(fitness3),3)))
            
                       )
        # self.x_std.set(np.std(fitness_df))
        # self.x_min.set(np.min(fitness_df))
        # self.x_max.set(np.max(fitness_df))
        # self.x_ind.set()
        
        n_gen = pd.DataFrame([algorithm.n_gen] * len(algorithm.pop), columns=[self.headers[0]])
        
        if(self.show_ind):
            # ind_df = [np.array2string(algorithm.pop.get("X").astype("int"),separator=',')]
            # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int),columns=[self.headers[-1]])

            # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int))
            with np.printoptions(linewidth=np.inf,threshold=np.inf):
                # ind_df = pd.DataFrame(algorithm.pop.get("X").astype(int))
                # STRING_BAD = '\n '
                individuals_list = list()
                
                for i in algorithm.pop.get("X").astype("int"):
                    string_array= np.array2string(i, separator=",") #.replace(STRING_BAD,"")
                    individuals_list.append(string_array)
                ind_df = pd.DataFrame(individuals_list, columns=[self.headers[-1]])
            # X_df
            output_df = pd.concat([n_gen,fitness_df,ind_df],axis=1)
        else:
            output_df = pd.concat([n_gen,fitness_df],axis=1)
        if(self.filepath is not None):        
            output_df.to_csv(self.filepath, mode="a", index=False, header=False) 


def saving_res_output(result,filepath_file=None, show_ind=False, display_verbose=False):
    """Given the pymoo result, display the fitness and (optional) the indiviual
        If filepath_file is different to None, it save it in the given filepath as csv. 

    Args:
        result (pymoo.core.result.Result): The result of the minimize problem
        filepath_file (str, optional):A filepath to save the output. Defaults to None.
        show_ind (bool, optional): Save/Display the indvidual representation. Defaults to False.
    """
    # Without this np.printoptions, the output is (therehold) summarize [1,0, ... 0,11]
    # and (linewidth) added \n 
    with np.printoptions(linewidth=np.inf,threshold=np.inf):
        n_gen = pd.DataFrame([result.algorithm.n_gen]*(result.F.shape[0]),columns=["gen"])
        fitness_df = pd.DataFrame(result.F, columns=["fitness1","fitness2","fitness3"])
        if(show_ind):
            # ind_df = pd.DataFrame([[str(x)] for x in result.X.astype(int)],columns=["Individuo"])
            # # STRING_BAD = '\n '
            individuals_list = list()
            for i in result.X.astype(int):
                # print(i)
                # string_array= np.array2string(i, separator=",", max_line_width=np.inf, threshold=sys.maxsize).replace(STRING_BAD,"")
                string_array= np.array2string(i, separator=",") #, max_line_width=np.inf, threshold=sys.maxsize)
                individuals_list.append(string_array)
            ind_df = pd.DataFrame(individuals_list, columns=["Individuo"])
            output_df = pd.concat([n_gen,fitness_df,ind_df],axis=1)
        else:
            output_df = pd.concat([n_gen,fitness_df],axis=1)
        if(filepath_file is not None):
            output_df.to_csv(filepath_file, index=False)
        if(display_verbose):
            print(output_df)
        
        # Return output
        # return output_df 



if __name__ == "__main__":
    """
    Pure Random Search baseline.

    For each generation, exactly POB new binary solutions are generated and
    evaluated. There is no crossover, mutation, mating, survival selection or
    duplicate elimination. pymoo is used for the Problem, Population,
    Evaluator, parallel elementwise evaluation and non-dominated sorting.
    """

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-s", "-seed", type=int, help='Seed for random search', required=True)
    parser.add_argument("-POB", "-MU", "-poblacion", "-pob", type=int,
                        help='Number of random solutions generated per generation',
                        required=False, default=4)
    parser.add_argument("-GEN", "-generacion", "-NGEN", "-gen", type=int,
                        help='Number of random-search generations',
                        required=False, default=5)
    parser.add_argument("-CPUS", "-cpus", "-hilos", type=int,
                        help='Number of CPUs used to evaluate solutions',
                        required=False, default=4)
    parser.add_argument("-v", "-verbose", "-show_pareto_sol", action='store_true',
                        required=False,
                        help='Save Pareto solutions including their binary individuals')
    parser.add_argument("-show_ind", action='store_true', required=False,
                        help='Include each binary individual in record_data CSV')
    parser.add_argument("-map", "-MAP", "-m", type=str,
                        choices=['Malaga', 'Melilla', 'malaga', 'melilla'],
                        default='Malaga', required=False,
                        help='Scenario/map to evaluate')
    parser.add_argument("-suffix", required=False, help='Change suffix folder result')

    start_time_stamp = time.time()
    print(f'Started at {time.strftime("%H:%M:%S %d-%m-%Y", time.gmtime(start_time_stamp))}')

    args = parser.parse_args()
    arguments = vars(args)
    print(arguments, flush=True)

    SEED = arguments["s"]
    NGEN = arguments["GEN"]
    MU = arguments["POB"]
    cpus = arguments["CPUS"]
    show_pareto = arguments["v"]
    show_ind = arguments["show_ind"]

    if NGEN <= 0:
        parser.error("-GEN must be greater than 0")
    if MU <= 0:
        parser.error("-POB must be greater than 0")
    if cpus <= 0:
        parser.error("-CPUS must be greater than 0")

    # Keep both RNGs seeded, as in the original experiment script.
    random.seed(SEED)
    np.random.seed(seed=SEED)
    rng = np.random.default_rng(SEED)

    scenario = arguments["map"].upper()
    if scenario == "MALAGA":
        SCENARIO = MALAGA_SCENARIO
        MAP_INPUT_FILE = MALAGA_INPUT_FILE
    else:
        SCENARIO = MELILLA_SCENARIO
        MAP_INPUT_FILE = MELILLA_INPUT_FILE

    base_path = '.'
    data_path = base_path + '/data-osm/' + SCENARIO + '/'

    suffix = arguments["suffix"]
    if suffix is None:
        suffix = ""

    results_path = base_path + "/results/" + SCENARIO + "/" + suffix + "/results-multiobjective/"
    images = results_path + "images/"
    map_path = data_path + "/maps/"

    Path(results_path).mkdir(parents=True, exist_ok=True)
    Path(images).mkdir(parents=True, exist_ok=True)
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(map_path).mkdir(parents=True, exist_ok=True)

    DISTRICT_DATA_FILE = f'districts-{SCENARIO}-data-with-nodes.csv'
    districts_data_df = pd.read_csv(data_path + DISTRICT_DATA_FILE)
    sections = districts_data_df['node'].tolist()

    filename_pair_list = "pair_less_than_3600_new_points.csv"
    pair_list_df = pd.read_csv(data_path + filename_pair_list)
    pair_list_df_selected = pair_list_df.loc[pair_list_df["sections"].isin(sections)]
    # Preserve the behavior of the base script, which ultimately uses all pairs.
    pair_list_df_selected = pair_list_df

    LIST_PAIR_OD = pair_to_list_tuples(pair_list_df_selected)
    print(len(LIST_PAIR_OD))

    # Search-space/map loading: identical to the base script.
    G_gpkg_nodes = gdp.read_file(map_path + f'{MAP_INPUT_FILE}.gpkg', layer="nodes").set_index("osmid")
    G_gpkg_edges = gdp.read_file(map_path + f'{MAP_INPUT_FILE}.gpkg', layer="edges").set_index(['u', 'v', 'key'])
    G = ox.graph_from_gdfs(G_gpkg_nodes, G_gpkg_edges)

    SCOOTER_SPEED = 2.78  # m/sec <- 10 KM/h
    WALKING_SPEED = 1.25  # m/sec <- 4.5 KM/h

    list_drive_weird = []
    num_lanes_total = 0
    count_subway = -1
    value_error_list = []

    num_lanes_great_eq_2 = 0
    dict_of_search_space_osmid = {}
    dict_of_search_space_u_v = {}
    dict_of_search_space_cost = {}
    dict_of_search_space_uvk_to_pos = {}

    for u, v, k, data in G.edges(data=True, keys=True):
        try:
            data["edge_walk"] = bool(data["edge_walk"])
            data["edge_scooter"] = bool(data["edge_scooter"])
            data["edge_subway"] = bool(data["edge_subway"])
            data["edge_drive"] = bool(data["edge_drive"])

            data["search_space"] = False
            data["edge_changed"] = False

            data["time_w"] = float(data["time_w"])
            data["time_w_c"] = float(data["time_w_c"])
            data[TIME_WEIGHT] = float(data[TIME_WEIGHT])
            data["time_w_m"] = float(data["time_w_m"])
            data['length'] = float(data['length'])

            try:
                data["osmid"] = int(data["osmid"])
            except Exception:
                data["osmid"] = count_subway
                count_subway -= 1

            if data["edge_drive"]:
                num_lanes_total += 1
                if not data["edge_scooter"]:
                    try:
                        data["lanes"] = int(data["lanes"])
                        if data["lanes"] >= 2:
                            if data["highway"] not in ["motorway", "motorway_link"]:
                                data["time_experiment_new"] = data['length'] / SCOOTER_SPEED
                                data["cost"] = 10
                                dict_of_search_space_uvk_to_pos[(u, v, k)] = num_lanes_great_eq_2
                                dict_of_search_space_osmid[num_lanes_great_eq_2] = data["osmid"]
                                dict_of_search_space_u_v[num_lanes_great_eq_2] = (u, v, k)
                                try:
                                    dict_of_search_space_cost[num_lanes_great_eq_2] = float(data['length'])
                                except Exception:
                                    print("Error en incorporar el length en (u=", u, ",v=", v, ")")

                                num_lanes_great_eq_2 += 1
                                data["search_space"] = True
                    except KeyError:
                        data["drive_weird"] = True
                        list_drive_weird.append((u, v, k, data))
                    except ValueError:
                        value_error_list.append((u, v, data["lanes"]))
                        data["lanes"] = 0

        except KeyError:
            print(data)
            print(KeyError)
            print("Error in change the type")
            print(f'u={u},v={v}')

    print(len(dict_of_search_space_osmid))
    print(len(dict_of_search_space_u_v))

    LEN_SEARCH_SPACE = num_lanes_great_eq_2
    NDIM = LEN_SEARCH_SPACE

    time_stamp = time.time()
    filename = '-PYMOO-{}-{}-{}-{}-{}-RANDOM_SEARCH-{}-{}'.format(
        SCENARIO,
        SEED,
        NGEN,
        MU,
        cpus,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time_stamp)),
        sys.platform
    )

    file_path = results_path + 'record_data-' + filename + '.csv'
    print(file_path, flush=True)

    if show_ind:
        headers = ["ngen", "fitness1", "fitness2", "fitness3", "Individuo"]
    else:
        headers = ["ngen", "fitness1", "fitness2", "fitness3"]

    # Initialize record_data with the same schema as FileOutput.
    pd.DataFrame({}, columns=headers).to_csv(file_path, index=False)

    all_X = []
    all_F = []

    print(
        f'Random Search: POB={MU}, GEN={NGEN}, total evaluations={MU * NGEN}, '
        f'CPUS={cpus}, seed={SEED}',
        flush=True
    )

    with multiprocessing.Pool(cpus) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = VectorOneMinimizeParallel(
            elementwise_runner=runner,
            list_pair_od=LIST_PAIR_OD,
            scenario=scenario,
            size_vector=LEN_SEARCH_SPACE
        )

        import pickle

        for gen in range(1, NGEN + 2):
            gen_start = time.time()

            # Exactly POB independent binary solutions. No mutation/crossover.
            X = rng.integers(0, 2, size=(MU, LEN_SEARCH_SPACE), dtype=np.int8)
            pop = Population.new("X", X)
            Evaluator().eval(problem, pop)

            F = np.asarray(pop.get("F"), dtype=float)
            X_eval = np.asarray(pop.get("X"), dtype=int)

            all_X.append(X_eval.copy())
            all_F.append(F.copy())

            fitness_df = pd.DataFrame(F, columns=["fitness1", "fitness2", "fitness3"])
            n_gen_df = pd.DataFrame([gen] * len(pop), columns=["ngen"])

            if show_ind:
                with np.printoptions(linewidth=np.inf, threshold=np.inf):
                    individuals_list = [np.array2string(i, separator=",") for i in X_eval]
                ind_df = pd.DataFrame(individuals_list, columns=["Individuo"])
                output_df = pd.concat([n_gen_df, fitness_df, ind_df], axis=1)
            else:
                output_df = pd.concat([n_gen_df, fitness_df], axis=1)

            output_df.to_csv(file_path, mode="a", index=False, header=False)

            # Preserve the first-population pickle artifact from the base script.
            if gen == 1:
                with open(results_path + 'population-' + filename + '.pkl', 'wb') as f:
                    print("Saving First population to pickle", flush=True)
                    pickle.dump(pop, f)

            mean_f = np.mean(F, axis=0)
            std_f = np.std(F, axis=0)
            min_f = np.min(F, axis=0)
            max_f = np.max(F, axis=0)
            print(
                f"Gen {gen}: pop size = {len(pop)} | "
                f"mean={tuple(np.round(mean_f, 3))} | "
                f"std={tuple(np.round(std_f, 3))} | "
                f"min={tuple(np.round(min_f, 3))} | "
                f"max={tuple(np.round(max_f, 3))} | "
                f"seconds={time.time() - gen_start:.3f}",
                flush=True
            )

    all_X = np.vstack(all_X)
    all_F = np.vstack(all_F)

    # Random Search has no survivor population. Its useful final result is the
    # non-dominated archive over every sampled/evaluated solution.
    pareto_idx = NonDominatedSorting().do(all_F, only_non_dominated_front=True)
    pareto_X = all_X[pareto_idx]
    pareto_F = all_F[pareto_idx]

    class RandomSearchResult:
        pass

    class RandomSearchAlgorithmState:
        pass

    results = RandomSearchResult()
    results.X = pareto_X
    results.F = pareto_F
    results.algorithm = RandomSearchAlgorithmState()
    results.algorithm.n_gen = NGEN

    saving_res_output(results, results_path + 'pareto_front' + filename + '.txt')
    if show_pareto:
        saving_res_output(
            results,
            results_path + 'pareto_front' + filename + '-show_ind.csv',
            show_ind=True
        )

    end_time_stamp = time.time()
    print("fin de Main\n", flush=True)
    print(
        '\n\nProcessed Random Search POB {} in NGEN {} ({} evaluations) in {} seconds. '
        'CPUS {} seed {}\n'.format(
            MU,
            NGEN,
            MU * NGEN,
            float(end_time_stamp - start_time_stamp),
            cpus,
            SEED
        ),
        flush=True
    )
    print("\n-----END OF THE JOB-----------\n", flush=True)
    print("\n-----END OF THE JOB-----------\n", flush=True, file=sys.stderr)
