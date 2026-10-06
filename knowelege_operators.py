
import numpy as np
import copy

import osmnx as ox
import networkx as nx

from pymoo.core.mutation import Mutation
from mutation import MultimodalMutation, ConnectivityMutationProbabilityDivided, DemandMutation

class KnowledgeMOD(MultimodalMutation):
    """KnowledgeMOD
        Consider the edge that gives access to public system as well as other cicleway as 1,
        other it is a probability if it changes
    """
    def _do(self, problem, X, prob_var, **kwargs):
        X = np.asarray(X)

        # prob_var = self.prob_var_value
       
        def is_nearby_bus(self, individual):
            
            # print((individual+self.ind_bus_stop_list)>1,flush=True)
            # print((individual+self.ind_bus_stop_list),flush=True)
            return ((individual + self.ind_bus_stop_list))
        
        def is_nearby_subway_stop(self, individual):
            return ((individual + self.ind_subway_stop_list))
        
        def is_nearby_cicleway(self, individual):
            return ((individual + self.adjency_scooter))
    
        def elementwiseDo(individual, prob_var):
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

        if X.ndim == 1:
            return elementwiseDo(X, prob_var=prob_var)
        return np.apply_along_axis(elementwiseDo, 1, X, prob_var=prob_var)


class KnowledgeCON(ConnectivityMutationProbabilityDivided):
    """KnowledgeCON
        Depending of a probability,
                If there is a full adjency (predecessor and succesor) for edge and prob<0.9, It mutates to one .
                If there no adjency for edge and prob<0.9, It mutates to zero .
                If there is a partial adjency and prob<0.6, It mutate to one
                Otherwise, flip
    """
    
    def _do(self, problem, X, prob_var, **kwargs):
        X = np.asarray(X)
        # prob_var_2 = self.get_prob_var(problem, size=(len(X), 1))
        # prob_var = self.prob_var_value
        # condition_mutation = np.random.random(X.shape) < prob_var
       
        def hasPrecessor(self, individual):
            # print(individual + self.precesors_base_array)
            return (individual + self.precesors_base_array)
        
        def hasSuccesor(self, individual):
            return (individual + self.succesors_base_array)
    
        def elementwiseDo(individual, verbose=False, prob_var=0.5):
            # flip = np.random.random(len(individual)) < [prob_var]*len(individual)
            Xp = copy.deepcopy(individual)
            succ_list = hasSuccesor(self, individual)
            pred_list = hasPrecessor(self, individual)
            for i in range(0,len(individual)):
                if np.random.random_sample() > prob_var:
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
                    
                if found_pred and verbose : print("Encontré predecesor")
                if found_succ and verbose: print("Encontré sucessor")
                if found_succ_neighbor and verbose: print("Encontré sucesor en mi vecino")
                if found_pred_neighbor  and verbose: print("Encontré predecesor en mi vecino")

            return Xp


        if X.ndim == 1:
            return elementwiseDo(X, prob_var=prob_var)
        return np.apply_along_axis(elementwiseDo, 1, X, prob_var=prob_var)



class KnowledgeDEM(DemandMutation):
    """KnowledgeDEM
        Calculate the use of each cicleway using "ALL" solution.
        Then, the probability of mutation to one is proportional to the
        use of the edge for the route, previously calculated
    """
    def _do(self, problem, X, prob_var, **kwargs):
            X = np.asarray(X)
            # prob_var = 
            # # print()
            # # a_zip = zip(prob_var,self.usage_list)
            # self.prob_var_ind = prob_var*np.array(self.usage_list)
                   
            
            def elementwiseDo(individual,prob_var):
                
                
                Xp = copy.deepcopy(individual)
                
                for i in range(0,len(individual)):
                    if np.random.random_sample() < prob_var:
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
    
            if X.ndim == 1:
                return elementwiseDo(X, prob_var)
            return np.apply_along_axis(elementwiseDo, 1, X, prob_var)
