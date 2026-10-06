
import numpy as np
import copy

import osmnx as ox
import networkx as nx

from pymoo.core.mutation import Mutation


def matched_random_mutation(individual: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """
    Matched Random Mutation:
    - Cuenta cuántos 0->1 y 1->0 hay entre Xp e individual.
    - Aplica esa misma cantidad de flips, pero en posiciones aleatorias del individuo.
    
    Parameters
    ----------
    individual : np.ndarray
        Vector binario (0/1) del individuo actual.
    Xp : np.ndarray
        Vector binario (0/1) que representa el cambio de referencia.
    rng : np.random.Generator
        Generador de aleatoriedad (p.ej. np.random.default_rng(seed)).
    
    Returns
    -------
    np.ndarray
        Nuevo individuo mutado (copia, no modifica el original).
    """

    original_hamming = np.count_nonzero(Xp != individual)

    # Diferencias por tipo
    diff_0_1_mask = (Xp == 0) & (individual == 1)  # donde Xp=0, ind=1  -> contamos 1->0 deseados
    diff_1_0_mask = (Xp == 1) & (individual == 0)  # donde Xp=1, ind=0  -> contamos 0->1 deseados

    n_0_to_1 = np.count_nonzero(diff_1_0_mask)  # número de 0->1 que queremos aplicar
    n_1_to_0 = np.count_nonzero(diff_0_1_mask)  # número de 1->0 que queremos aplicar

    # Índices disponibles en el individuo
    idx_0 = np.flatnonzero(individual == 0)
    idx_1 = np.flatnonzero(individual == 1)

    Xp_new = individual.copy()

    # Aplicar 0 -> 1
    if n_0_to_1 > 0 and idx_0.size > 0:
        k = min(n_0_to_1, idx_0.size)
        chosen = np.random.choice(idx_0, size=k, replace=False)
        Xp_new[chosen] = 1

    # Aplicar 1 -> 0
    if n_1_to_0 > 0 and idx_1.size > 0:
        k = min(n_1_to_0, idx_1.size)
        chosen = np.random.choice(idx_1, size=k, replace=False)
        Xp_new[chosen] = 0

        # Distancia de Hamming después de la mutación
    new_hamming = np.count_nonzero(individual != Xp_new)
    
    # Verificar que la distancia de Hamming se preserva
    assert original_hamming == new_hamming, \
        f"La distancia de Hamming no se preservó: {original_hamming} -> {new_hamming}"
    
    return Xp_new


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
        
        # print("Multimodalidad-Init->Making list and dictionaries",flush=True)
        for i in range(0,len(self.all_on_individual)):         
            u,v,k = self.dict_of_search_space_u_v[i]  
            
            ## Check for each position, if there is a bus stop nearby and save the answer in a list    
            bus_stop_nearby_edges = self.map.edges(nbunch=[u,v], data="type_edge_bus", default=None, keys=True)
            bus_stop_nearby_list = [True if (data_verify and ("bus_stop" in data_verify)) else False for no, ne, nk, data_verify in bus_stop_nearby_edges]
            cond_bus_stop = any(bus_stop_nearby_list)
            self.ind_bus_stop_list[i] = cond_bus_stop
            
            ## Check for each position, if there is a subway stop nearby(connected to u or v) and save the answer in a list
            # subway_stop_nearby_edges = self.map.edges(nbunch=[u,v], data="type_subway_edge", default=None, keys=True)
            # subway_stop_nearby_edges_list = [True if data_verify and ("stop" in data_verify ) else False for no, ne, nk, data_verify in subway_stop_nearby_edges]
            # cond_subway_stop = any(subway_stop_nearby_edges_list)
            subway_stop_nearby_edges = self.map.edges(nbunch=[u,v], data="edge_subway", default=None, keys=True)
            subway_stop_nearby_edges_list = [True if data_verify  else False for no, ne, nk, data_verify in subway_stop_nearby_edges]
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
        # print("Multimodalidad-Init->End making dictionaries",flush=True)
        self.prob_var_value = prob_var     
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):

        prob_var = self.prob_var_value
       
        def is_nearby_bus(self, individual):
            
            # # print((individual+self.ind_bus_stop_list)>1,flush=True)
            # # print((individual+self.ind_bus_stop_list),flush=True)
            return ((individual + self.ind_bus_stop_list))
        
        def is_nearby_subway_stop(self, individual):
            return ((individual + self.ind_subway_stop_list))
        
        def is_nearby_cicleway(self, individual):
            return ((individual + self.adjency_scooter))
    
        def elementwiseDo(individual):
            # print("Multimodalidad-elementWiseDo->PRE",flush=True)
            Xp = copy.deepcopy(individual)
            nearby_bus_list = is_nearby_bus(self, individual)
            nearby_subway_list = is_nearby_subway_stop(self, individual)
            nearby_cycleway_list = is_nearby_cicleway(self, individual)
            # print("Antes del Bucle->PRE",flush=True)
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
                    ## print("Encontre Adyancentes",flush=True)
                else:
                    ## print("NO Encontre Adyancentes",flush=True)
                    if (np.random.random_sample() < prob_var):
                        Xp[i] = ~individual[i]
            # if found: # print("Encontré algún adyacente", flush=True)
            # if entered: # print("Encontré algún vecino", flush=True)
            # print("Multimodalidad-elementWiseDo->End",flush=True)
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
            #     # print(f"origen={no} predecessor de {u}")
            #     # print(f"destino={ne} u={u}")
            #     # print(ne != u)
            #     # print(data_verify)
            cond = any(pred_es)
            self.precesors_base_array[i] = cond
        self.prob_var_value = prob_var
        
        
            
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):
        prob_var_2 = self.get_prob_var(problem, size=(len(X), 1))
        prob_var = self.prob_var_value
        # condition_mutation = np.random.random(X.shape) < prob_var
       
        def hasPrecessor(self, individual):
            # # print(individual + self.precesors_base_array)
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
                else:      
                    Xp[i] = ~individual[i]
                    
                # if found_pred and verbose : # print("Encontré predecesor")
                # if found_succ and verbose: # print("Encontré sucessor")
                # if found_succ_neighbor and verbose: # print("Encontré sucesor en mi vecino")
                # if found_pred_neighbor  and verbose: # print("Encontré predecesor en mi vecino")

            return Xp


        return np.apply_along_axis(elementwiseDo, 1,X)

class DemandMutation(Mutation):
    """DemandMutation
        Calculate the use of each cicleway using "ALL" solution.
        Then, the probability of mutation to one is proportional to the
        use of the edge for the route, previously calculated
    """
    
    def __init__(self, graph, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list,prob_var):
        # print("mutation=>MUTUSAGE_FLIP",flush=True)
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
        
        # # print(f"There any usage>1 {any([1 if x>1 else 0 for x in self.usage_list])}")
        # 0.5 + (x/#routes)
        # self.usage_list = list(map(lambda x : 0.5 + (x/len(pair_list)),self.usage_list))
        self.prob_vector_list = list(map(lambda x: 0 if x == 0 else 0.5 + (x/self.number_edges), self.usage_list))
        self.prob_var_value = prob_var        
        self.prob_vector = np.array(self.prob_vector_list)
            
        super().__init__(prob_var=prob_var)

    def _do(self, problem, X, **kwargs):
        # prob_var = 
        # # # print()
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




class ActionMatchedRandomMutation_Connectivity(ConnectivityMutationProbabilityDivided):
    """ActionMatchedRandomMutation_Connectivity
        Hamming Distance mutation using the information of ConnectivityMutation
    """
    # __init__ from ConnectivityMutationProbabilityDivided

    def _do(self, problem, X, **kwargs):
        prob_var_2 = self.get_prob_var(problem, size=(len(X), 1))
        prob_var = self.prob_var_value
        # condition_mutation = np.random.random(X.shape) < prob_var
       
        def hasPrecessor(self, individual):
            # # print(individual + self.precesors_base_array)
            return (individual + self.precesors_base_array)
        
        def hasSuccesor(self, individual):
            return (individual + self.succesors_base_array)
    
        def elementwiseDo(individual,verbose=False):
            # flip = np.random.random(len(individual)) < [prob_var]*len(individual)
            Xp = copy.deepcopy(individual)
            succ_list = hasSuccesor(self, individual)
            pred_list = hasPrecessor(self, individual)

            matching_one = []
            matching_zero = []
            
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
                else:      
                    Xp[i] = ~individual[i]
                    
                # if found_pred and verbose : # print("Encontré predecesor")
                # if found_succ and verbose: # print("Encontré sucessor")
                # if found_succ_neighbor and verbose: # print("Encontré sucesor en mi vecino")
                # if found_pred_neighbor  and verbose: # print("Encontré predecesor en mi vecino")


            Xp_new = matched_random_mutation(individual, Xp)
            
            # # return  Xp # Intead let's do a matching
            # diff_0_1 = [i for i in range(len(Xp)) if (Xp[i] == 0 and individual[i] == 1)]
            # diff_1_0 = [i for i in range(len(Xp)) if (Xp[i] == 1 and individual[i] == 0)]

            # number_diff_0_1 = len(diff_0_1)
            # number_diff_1_0 = len(diff_1_0)
            # Xp_new = copy.deepcopy(individual)
            # # Debo cambiar de forma aleatoria number_diff_0_1   elementos de 0 a 1 en individual
            # individual_0 = [i for i in range(len(individual)) if individual[i] == 0]
            # individual_1 = [i for i in range(len(individual)) if individual[i] == 1]
            # if number_diff_0_1 > 0:
            #     position_to_flip =np.random.choice(individual_0, size=number_diff_0_1, replace=False)
            #     for pos in position_to_flip:
            #         Xp_new[pos] = 1
            # if number_diff_1_0 > 0:
            #     position_to_flip =np.random.choice(individual_1, size=number_diff_1_0, replace=False)
            #     for pos in position_to_flip:
            #         Xp_new[pos] = 0


            return Xp_new


        return np.apply_along_axis(elementwiseDo, 1,X)


class ActionMatchedRandomMutation_Demand(DemandMutation):
    """ActionMatchedRandomMutation_Demand
        Hamming Distance mutation using the information of DemandMutation
    """    
    def _do(self, problem, X, **kwargs):
        # prob_var = 
        # # # print()
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

            Xp_new = matched_random_mutation(individual, Xp)
            return Xp_new

        return np.apply_along_axis(elementwiseDo, 1,X)

class ActionMatchedRandomMutation_Multimodal(MultimodalMutation):
    """ActionMatchedRandomMutation_Multimodal
        Hamming Distance mutation using the information of MultimodalMutation
    """
    def _do(self, problem, X, **kwargs):
        prob_var = self.prob_var_value
       
        def is_nearby_bus(self, individual):
            
            # # print((individual+self.ind_bus_stop_list)>1,flush=True)
            # # print((individual+self.ind_bus_stop_list),flush=True)
            return ((individual + self.ind_bus_stop_list))
        
        def is_nearby_subway_stop(self, individual):
            return ((individual + self.ind_subway_stop_list))
        
        def is_nearby_cicleway(self, individual):
            return ((individual + self.adjency_scooter))
    
        def elementwiseDo(individual):
            # print("Multimodalidad-elementWiseDo->PRE",flush=True)
            Xp = copy.deepcopy(individual)
            nearby_bus_list = is_nearby_bus(self, individual)
            nearby_subway_list = is_nearby_subway_stop(self, individual)
            nearby_cycleway_list = is_nearby_cicleway(self, individual)
            # print("Antes del Bucle->PRE",flush=True)
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
                    ## print("Encontre Adyancentes",flush=True)
                else:
                    ## print("NO Encontre Adyancentes",flush=True)
                    if (np.random.random_sample() < prob_var):
                        Xp[i] = ~individual[i]
            # if found: # print("Encontré algún adyacente", flush=True)
            # if entered: # print("Encontré algún vecino", flush=True)
            # print("Multimodalidad-elementWiseDo->End",flush=True)
            # return Xp

            # print("##########################")
            # print("Matched Random Mutation")
            Xp_new = matched_random_mutation(individual, Xp)
            # print("End Matched Random Mutation")
            return Xp_new

        return np.apply_along_axis(elementwiseDo, 1,X)


class CombinatedMutation(Mutation):
    """
    CombinatedMutation:
        Depending of a probability, it uses one of the following mutations:
            - ConnectivityMutationProbabilityDivided
            - DemandMutation
            - MultimodalMutation
    The mutation operators are passed as a list of tuples (mutation_instance, probability),
    and should be already initialized with their specific parameters.
    
    Example:
    --------
    >>> conn_mut = ConnectivityMutationProbabilityDivided(graph=G, ...)
    >>> demand_mut = DemandMutation(graph=G, pair_list=pairs, ...)
    >>> multimodal_mut = MultimodalMutation(graph=G, ...)
    >>> 
    >>> combinated_mut = CombinatedMutation([
    ...     (conn_mut, 0.4),
    ...     (attention_mut, 0.3),
    ...     (multimodal_mut, 0.3)
    ... ])
    """
    
    def __init__(self, mutation_operators, verbose=True, seed=None):
        """
        Parameters
        ----------
        mutation_operators : list of tuples
            List of tuples (mutation_instance, probability)
            Example: [(conn_mut, 0.4), (attention_mut, 0.3), (multimodal_mut, 0.3)]
            
        verbose : bool, default=True
            Si True, print name of operator
            
        seed : int, optional
            Seed para reproducibilidad
        """
        super().__init__()
        
        # Validar entrada
        if len(mutation_operators) == 0:
            raise ValueError("list cannot be empty")
        
        # Extraer operadores y probabilidades
        self.mutation_operators = [op for op, _ in mutation_operators]
        self.probs = [prob for _, prob in mutation_operators]
        self.names = [op.__class__.__name__ for op, _ in mutation_operators]
        
        # Validar que las probabilidades sumen 1
        total_prob = sum(self.probs)
        if not np.isclose(total_prob, 1.0, atol=0.01):
            raise ValueError(f"It should sum to 1.0, but sum to {total_prob}")
        
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def _do(self, problem, X, **kwargs):
        """
        Select one operator and applied to the whole population at X
        """
        # Seleccionar operador según probabilidades
        selected_idx = self.rng.choice(len(self.mutation_operators), p=self.probs)
        selected_operator = self.mutation_operators[selected_idx]
        
        # if self.verbose:
        #     print(f"CombinatedMutation: Using {self.names[selected_idx]}")
        
        # Aplicar a toda la población
        return selected_operator._do(problem, X, **kwargs)



class SwapMutation(Mutation):
    """
    SwapMutation:
        Randomly selects two positions in the individual and swaps their values.
        This mutation is applied with a given probability.
    """
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # Xp = np.copy(X)
        def elementwiseDo(individual):
            individual_b = np.copy(individual)
            if np.random.rand() < self.prob:
                n_var = len(individual)
                i, j = np.random.choice(n_var, size=2, replace=False)
                individual[i], individual[j] = individual[j], individual[i]
                hamming_distance = np.count_nonzero(individual_b != individual)
                assert ((hamming_distance == 2) or (hamming_distance == 0)), f"Hamming distance should be 2 or 0 after swap, got {hamming_distance}"
            return individual
        return np.apply_along_axis(elementwiseDo, 1, X)