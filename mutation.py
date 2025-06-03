
import numpy as np
import copy

import osmnx as ox
import networkx as nx

from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp
        
class MultimodalMutation(Mutation):
    """MultimodalMutation
        Consider the edge that gives access to public system as well as other cicleway as 1,
        other it is a probability if it changes
    """
    def __init__(self, graph, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,prob_var,weight_metric="time_experiment"):
        self.map = (graph)
        self.dict_of_search_space_u_v = dict_of_search_space_u_v
        self.dict_of_search_space_uvk_to_pos = dict_of_search_space_uvk_to_pos
        
        
        
        self.ind_bus_stop_list = [0]*LEN_SEARCH_SPACE 
        self.ind_subway_stop_list = [0]*LEN_SEARCH_SPACE 
        self.adjency_scooter = [0]*LEN_SEARCH_SPACE 
        
        self.adjency_change_scooter = {} # When edge changed
        
        
        self.all_on_individual = [1]*LEN_SEARCH_SPACE 
        bounded = copy.deepcopy(self.map)
        for i in range(0,LEN_SEARCH_SPACE):
            if(self.all_on_individual[i]):
                u,v,k = dict_of_search_space_u_v[i]
                data = bounded.get_edge_data(u, v, key=k)
                if(data == None):
                    print("We have a situation here")
                else:
                    data["time_w_c"] = data["time_experiment_new"]
                    data[weight_metric] = data["time_experiment_new"]
                    data["edge_changed"] = True
        
        print("Multimodalidad-Init->Making list and dictionaries",flush=True)
        for i in range(0,len(self.all_on_individual)):         
            u,v,k = self.dict_of_search_space_u_v[i]  
            
            ## Check for each position, if there is a bus stop nearby and save the answer in a list    
            bus_stop_nearby_edges = self.map.edges(nbunch=[u,v], data="type_edge_bus", default=None, keys=True)
            bus_stop_nearby_list = [True if (data_verify and ("bus_stop" in data_verify)) else False for no, ne, nk, data_verify in bus_stop_nearby_edges]
            cond_bus_stop = any(bus_stop_nearby_list)
            self.ind_bus_stop_list[i] = cond_bus_stop
            
            ## Check for each position, if there is a subway stop nearby(connected to u or v) and save the answer in a list
            subway_stop_nearby_edges = self.map.edges(nbunch=[u,v], data="type_subway_edge", default=None, keys=True)
            subway_stop_nearby_edges_list = [True if data_verify and ("stop" in data_verify ) else False for no, ne, nk, data_verify in subway_stop_nearby_edges]
            cond_subway_stop = any(subway_stop_nearby_edges_list)
            self.ind_subway_stop_list[i] = cond_subway_stop
 
            ## Check for each node if there is any edge_scooter (cicleway) nearby
            adjency_scooter_nearby = self.map.in_edges(nbunch=[u,v], data="edge_scooter", default=False, keys=True)
            adjency_scooter_list = [True if data_verify else False for no, ne, nk, data_verify in adjency_scooter_nearby]
            cond_scooter_adjency = any(adjency_scooter_list)
            self.adjency_scooter[i] = cond_scooter_adjency
            
            
        
            ## Check if this nearby when it changed
            self.adjency_change_scooter[i] = list()
               
            succesors_edc = bounded.edges(nbunch=v, data="edge_changed", keys=True)
            succ_connected_list = [ self.dict_of_search_space_uvk_to_pos[(no,ne,nk)] if ne != u and data_verify else None for no, ne, nk, data_verify in succesors_edc]
 
            precessor_edc = bounded.in_edges(nbunch=u, data="edge_changed", keys=True)
            pred_connected_list = [self.dict_of_search_space_uvk_to_pos[(no,ne,nk)] if ne != v and data_verify else None for no, ne, nk, data_verify in precessor_edc]
            
            self.adjency_change_scooter[i] = succ_connected_list+pred_connected_list  
        print("Multimodalidad-Init->End making dictionaries",flush=True)
        self.prob_var_value = prob_var     
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):

        prob_var = self.prob_var_value
       
        def is_nearby_bus(self, individual):
            
            # print((individual+self.ind_bus_stop_list)>1,flush=True)
            # print((individual+self.ind_bus_stop_list),flush=True)
            return ((individual + self.ind_bus_stop_list))
        
        def is_nearby_subway_stop(self, individual):
            return ((individual + self.ind_subway_stop_list))
        
        def is_nearby_cicleway(self, individual):
            return ((individual + self.adjency_scooter))
    
        def elementwiseDo(individual):
            print("Multimodalidad-elementWiseDo->PRE",flush=True)
            Xp = copy.deepcopy(individual)
            nearby_bus_list = is_nearby_bus(self, individual)
            nearby_subway_list = is_nearby_subway_stop(self, individual)
            nearby_cycleway_list = is_nearby_cicleway(self, individual)
            print("Antes del Bucle->PRE",flush=True)
            found = False
            entered = False
            for i in range(0,len(individual)):
                sumAdj = 0

                nearby_bus = nearby_bus_list[i] 
                nearby_cycleway = nearby_cycleway_list[i]
                nearby_subway = nearby_subway_list[i]
                
                # if (nearby_bus or nearby_cycleway or nearby_subway):
                #     # There is any bus, cycleway or subway near
                #     sumAdj = 1
                #     found = True     
                # else:
                #     # Not found any bus, cycleway or subway let's see
                #     # if there is a neighbor that is changed to e-scooter.
                #     list_index =  self.adjency_change_scooter[i]
                    
                #     for index_ in list_index:
                #         if index_ is not None:
                            
                #             if individual[index_]:
                #                 entered = True
                #                 # True if the neighbor changed in this iteration (then it is near to a cycleway)
                #                 sumAdj = 1
                #                 break
                
                if not (nearby_bus or nearby_cycleway or nearby_subway):
                   # Not found any bus, cycleway or subway let's see
                    # if there is a neighbor that is changed to e-scooter.
                    list_index =  self.adjency_change_scooter[i]
                    
                    for index_ in list_index:
                        if index_ is not None:
                            
                            if individual[index_]:
                                entered = True
                                # True if the neighbor changed in this iteration
                                # (then it is near to a cycleway)
                                sumAdj = 1
                                break      
                else:
                    # There is any bus, cycleway or subway near
                    sumAdj = 1
                    found = True
                    
                
                prob_condition = np.random.random_sample()
                if(sumAdj and prob_condition<prob_var):
                # if(sumAdj):
                    Xp[i] = 1
                    #print("Encontre Adyancentes",flush=True)
                else:
                    #print("NO Encontre Adyancentes",flush=True)
                    if (np.random.random_sample() < prob_var):
                        Xp[i] = ~individual[i]
            if found: print("Encontré algún adyacente", flush=True)
            if entered: print("Encontré algún vecino", flush=True)
            print("Multimodalidad-elementWiseDo->End",flush=True)
            return Xp

        return np.apply_along_axis(elementwiseDo, 1,X)


class ConnectivityMutationProbabilityDivided(Mutation):
    """ConnectivityMutationProbabilityDivided
        Depending of a probability,
                If there is a full adjency (predecessor and succesor) for edge and prob<0.9, It mutates to one .
                If there no adjency for edge and prob<0.9, It mutates to zero .
                If there is a partial adjency and prob<0.6, It mutate to one
                If there proba<0.3, flip
                Otherwise, keep the same
    """
    
    def __init__(self, graph, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,prob_var,weight_metric="time_experiment",verbose=True):
        self.map = (graph)
        self.dict_of_search_space_u_v = dict_of_search_space_u_v
        self.dict_of_search_space_uvk_to_pos = dict_of_search_space_uvk_to_pos
        self.precesors_connected_dict = {}
        self.succesors_connected_dict = {}
        self.succesors_base_array = [0]*LEN_SEARCH_SPACE 
        self.precesors_base_array = [0]*LEN_SEARCH_SPACE 
        self.all_on_individual = [1]*LEN_SEARCH_SPACE
        
        bounded = copy.deepcopy(self.map)
        for i in range(0,LEN_SEARCH_SPACE):
            if(self.all_on_individual[i]):
                u,v,k = dict_of_search_space_u_v[i]
                data = bounded.get_edge_data(u, v, key=k)
                if(data == None):
                    print("We have a situation here")
                else:
                    data["time_w_c"] = data["time_experiment_new"]
                    data[weight_metric] = data["time_experiment_new"]
                    data["edge_changed"] = True
                    
        for i in range(0,len(self.all_on_individual)):
            self.precesors_connected_dict[i] = list()
            self.succesors_connected_dict[i] = list()
            
            u,v,k = self.dict_of_search_space_u_v[i]       
            succesors_edc = bounded.out_edges(nbunch=v, data="edge_changed", keys=True)
            succ_connected_list = [ self.dict_of_search_space_uvk_to_pos[(no,ne,nk)] if ne != u and data_verify else None for no, ne, nk, data_verify in succesors_edc]
 
            precessor_edc = bounded.in_edges(nbunch=u, data="edge_changed", keys=True)
            pred_connected_list = [self.dict_of_search_space_uvk_to_pos[(no,ne,nk)] if ne != v and data_verify else None for no, ne, nk, data_verify in precessor_edc]
            self.succesors_connected_dict[i] = succ_connected_list
            self.precesors_connected_dict[i] = pred_connected_list
  
            succesors_es = self.map.out_edges(nbunch=v, data="edge_scooter", default=False, keys=True)           
            succ_es = [data_verify if ne != u else False for no, ne, nk, data_verify in succesors_es]
            cond = any(succ_es)
            self.succesors_base_array[i] = cond
        
            precessor_es = self.map.in_edges(nbunch=u, data="edge_scooter", default=False, keys=True)           
            pred_es = [data_verify if ne != v else False for no, ne, nk, data_verify in precessor_es]
            # for no, ne, nk, data_verify in precessor_es:
            #     print(f"origen={no} predecessor de {u}")
            #     print(f"destino={ne} u={u}")
            #     print(ne != u)
            #     print(data_verify)
            cond = any(pred_es)
            self.precesors_base_array[i] = cond
        self.prob_var_value = prob_var 
        
        
        
            
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):
        prob_var_2 = self.get_prob_var(problem, size=(len(X), 1))
        prob_var = self.prob_var_value
        # condition_mutation = np.random.random(X.shape) < prob_var
       
        def hasPrecessor(self, individual):
            # print(individual + self.precesors_base_array)
            return (individual + self.precesors_base_array)
        
        def hasSuccesor(self, individual):
            return (individual + self.succesors_base_array)
    
        def elementwiseDo(individual,verbose=False):
            # flip = np.random.random(len(individual)) < [prob_var]*len(individual)
            Xp = copy.deepcopy(individual)
            succ_list = hasSuccesor(self, individual)
            pred_list = hasPrecessor(self, individual)
            for i in range(0,len(individual)):
                if np.random.random_sample() > self.prob_var_value:
                    # Filp a coin and do not try to mutate this edge
                    continue
                sumSucc = 0
                sumPred = 0
                
                found_pred = False
                found_succ = False
                found_succ_neighbor = False
                found_pred_neighbor = False
                
                if not(succ_list[i]):
                    # Not found succesor, but let's explore
                    # if there those who changed are a succesor
                    # of i
                    list_index =  self.succesors_connected_dict[i]
                    for index_ in list_index:
                        if index_ is not None:
                            if individual[index_]:
                                # Uno de mis vecinos cambio
                                found_succ_neighbor = True
                                sumSucc = 1
                                break
                else:
                    # Found succesor in current
                    found_succ = True
                    sumSucc = 1
                    
                if not(pred_list[i]):
                    # Not found predecessor, but let's explore
                    # if there those who changed are a predecessor
                    # of i
                    list_index = self.precesors_connected_dict[i]            
                    for index_ in list_index:
                        if index_ is not None:
                            if individual[index_]:
                                # Uno de mis vecinos cambio
                                found_pred_neighbor = True
                                sumPred = 1
                                break
                else:
                    # found predecessor
                    sumPred = 1
                    found_pred = True
                
                sumAdj = sumSucc + sumPred
                prob_condition = np.random.random()
            
                if ((prob_condition < (0.9)) and (sumAdj == 2)):
                    # Connection in succesors and in precessors
                    Xp[i] = 1
                elif ((prob_condition < (0.9)) and (sumAdj == 0)):
                    Xp[i] = 0
                elif (prob_condition < (0.6) and (sumAdj == 1) ): 
                    # Connection in only one (sucessors or precessors)
                    Xp[i] = 1
                elif(prob_condition < (0.3)):      
                    Xp[i] = ~individual[i]
                    
                if found_pred and verbose : print("Encontré predecesor")
                if found_succ and verbose: print("Encontré sucessor")
                if found_succ_neighbor and verbose: print("Encontré sucesor en mi vecino")
                if found_pred_neighbor  and verbose: print("Encontré predecesor en mi vecino")

            return Xp


        return np.apply_along_axis(elementwiseDo, 1,X)

class AttentionUseMutationFlip(Mutation):
    """AttentionUseMutationFlip
        Calculate the use of each cicleway using "ALL" solution.
        Then, the probability of mutation to one is proportional to the
        use of the edge for the route, previously calculated
    """
    
    def __init__(self, graph, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list,prob_var):
        print("mutation=>MUTUSAGE_FLIP",flush=True)
        self.map = (graph)
        self.dict_of_search_space_u_v = dict_of_search_space_u_v
        self.dict_of_search_space_uvk_to_pos = dict_of_search_space_uvk_to_pos
        
        self.usage_list = [0]*LEN_SEARCH_SPACE
        
        self.all_on_individual = [1]*LEN_SEARCH_SPACE 
        
        # Changing map for all map
        bounded = copy.deepcopy(self.map)
        for i in range(0,LEN_SEARCH_SPACE):
            if(self.all_on_individual[i]):
                u,v,k = dict_of_search_space_u_v[i]
                data = bounded.get_edge_data(u, v, key=k)
                if(data == None):
                    print("We have a situation here")
                else:
                    data["time_w_c"] = data["time_experiment_new"]
                    data[weight_metric] = data["time_experiment_new"]
                    data["edge_changed"] = True
        
        self.number_edges = bounded.number_of_edges()          
        ## To know usage of the edges 
        for i in pair_list:
            orig_node = i[0]
            dest_node = i[1]
            try:
                # Map should be a directed graph with weight in non-negative integer
                _, route_of_nodes = nx.bidirectional_dijkstra(bounded,orig_node,dest_node, weight_metric) # get distance and routes
                route_edges_attr = ox.utils_graph.route_to_gdf(bounded, route_of_nodes,weight=weight_metric) # list of Edges attributes of route


                for edge in route_edges_attr.itertuples():
                    if(getattr(edge,"edge_changed")):
                        u_escooter,v_escooter,k_escooter = getattr(edge,"Index")
                        u_v_k_escooter = (u_escooter,v_escooter,k_escooter)
                        pos = self.dict_of_search_space_uvk_to_pos[u_v_k_escooter]
                        self.usage_list[pos] += 1
                    
            except Exception as error:
                print(f"An error ocurred:{type(error).__name__}")
                print(f"Details:\n{error}")
                print(f"\tError in pair_list->{pair_list}")
        
        # print(f"There any usage>1 {any([1 if x>1 else 0 for x in self.usage_list])}")
        # 0.5 + (x/#routes)
        # self.usage_list = list(map(lambda x : 0.5 + (x/len(pair_list)),self.usage_list))
        self.prob_vector_list = list(map(lambda x: 0 if x == 0 else 0.5 + (x/self.number_edges), self.usage_list))
        self.prob_var_value = prob_var        
        self.prob_vector = np.array(self.prob_vector_list)
            
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):
        # prob_var = 
        # # print()
        # # a_zip = zip(prob_var,self.usage_list)
        # self.prob_var_ind = prob_var*np.array(self.usage_list)
               
        
        def elementwiseDo(individual):
            
            
            Xp = copy.deepcopy(individual)
            
            for i in range(0,len(individual)):
                if np.random.random_sample() < self.prob_var_value:
                    if self.prob_vector[i] == 0:
                       Xp[i] = ~ Xp[i] 
                    else: # self.prob_vector[i] == 1:
                        newRand = np.random.random_sample()
                        prob_ind = self.prob_vector[i]
                        if(Xp[i]==0):
                            if (newRand<prob_ind):
                                Xp[i] = 1
                        else: # Xp[i] ==1
                            if (newRand>prob_ind):
                                Xp[i] = 0
            return Xp

        return np.apply_along_axis(elementwiseDo, 1,X)

