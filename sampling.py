import sys

import numpy as np
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from knowelege_operators import KnowledgeCON, KnowledgeDEM, KnowledgeMOD

## My BinaryOperator
class BinaryRandomSamplingCustom(BinaryRandomSampling):
    def __init__(self, df_individual) -> None:
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        self.individual_custom = df_individual
        super().__init__()
    
    # Custom approach. Erase the first individuals
    def _first_one(self, problem, n_samples):
        val = np.random.random((n_samples, problem.n_var))
        counter = 0
        for indv in self.individual_custom.itertuples():
            new_ind = eval(getattr(indv,"individuals"))
            val[counter] = new_ind
            counter += 1 
        return (val < 0.5).astype(bool)
        
        
    
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)




import pickle
class CombinationThreeKnowledgeSampling(BinaryRandomSampling):
    """
        This abstract class represents the initialization sampling
        that creates a initial population considering
        5 Kind of solutions: Connectivity, Demand, Multimodal, Random and Custom
        The Custom solutions are provided by the user in a dataframe with the column "individuals" containing the binary representation of the solution.

        n_samples: number of samples to generate
        problem.n_var: number of variables in the problem(Number of states (0,1) in the binary representation of the solution LEN_SEARCH_SPACE)    

        32 solutions will be generated in total, 8 random, 3 custom, 7 connectivity, 7 demand and 7 multimodal.

        8 will be generated randomly,
        3 will be generated from the custom solutions provided by the user,
        7 will be generated from the current solution and then applying the connectivity knowledge with a probablity of {1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1}
        7 will be generated from the current solution and then applying the demand knowledge with a probablity of {1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1}
        7 will be generated from the current solution and then applying the multimodal knowledge with a probablity of {1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1}
        
    """
    def __init__(self, df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list) -> None:
        
        self.individual_custom = df_individual
        
        MOD_Mutation = KnowledgeMOD(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment")
        CON_Mutation = KnowledgeCON(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment",verbose=True)
        DEM_Mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
        self.mutation_dict = {"MOD": MOD_Mutation, "CON": CON_Mutation, "DEM": DEM_Mutation}
        print("Mutation dict created")
        # self.mutation_dict = mutation_dict
        super().__init__()
        
    
    def _do(self, problem, n_samples):
        # val = np..random((n_samples, problem.n_var))
        if n_samples < 32:
            raise ValueError(f"n_samples must be at least 32, but got {n_samples}.")
        val = np.zeros((n_samples, problem.n_var), dtype=bool)
        # with open("val_before.pkl", "wb") as f:
        #     pickle.dump(val, f)

        shape_before = val.shape
        partitions = int(n_samples/4)
        rest = n_samples % 4
        CUSTOM_SOLUTIONS = 3
        if rest != 0:
            print(f"Warning: n_samples ({n_samples}) is not divisible by 4. The rest of the division will be added to the random partition.")

        solutions_for_knowledge = int(partitions - 1 )# for custom
        probablity_list = [i/(solutions_for_knowledge) for i in range(1, solutions_for_knowledge+1)]    

        # Random solutions
        solutions = 0
        for counter in range(0, (partitions+rest)):
            val[solutions] = (np.random.random(problem.n_var) < 0.5).astype(bool)
            solutions +=1
        # Custom solutions
        # with open("val_after_random.pkl", "wb") as f:
        #     pickle.dump(val, f)
                    
        # print("\t Generating CUSTOM knowledge solutions")
        for counter , indv in zip (range((partitions+rest), (partitions+rest+CUSTOM_SOLUTIONS)), self.individual_custom.itertuples()):
            new_ind = eval(getattr(indv,"individuals"))
            val[solutions] = new_ind
            counter += 1
            solutions +=1
        # with open("val_after_custom.pkl", "wb") as f:
        #     pickle.dump(val, f)
        # print("\tsolutions{}={}".format(counter,solutions))
        # # Connectivity knowledge solutions
        # print("\t Generating CON knowledge solutions")
        initial_counter = (partitions+rest+CUSTOM_SOLUTIONS)
        total_counter = initial_counter + (solutions_for_knowledge)
        for counter in range(initial_counter, total_counter):
            try:
                prob_var = probablity_list[solutions-initial_counter]
            except:
                print("CON", flush=True, file=sys.stderr)
                print(f"IndexError: solutions={solutions}, initial_counter={initial_counter}, len(probablity_list)={len(probablity_list)}", flush=True)
                print(f"solutions-initial_counter={solutions-initial_counter}", flush=True)
                raise
            indv = val[solutions].reshape(1, -1)
            
            mutation = self.mutation_dict.get("CON")            
            val[solutions] = mutation._do(problem, X=indv, prob_var=prob_var)
            solutions +=1
        # with open("val_after_connectivity.pkl", "wb") as f:
        #     pickle.dump(val, f)
        # print("\tsolutions{}={}".format(counter,solutions))
        # # Demand knowledge solutions
        # print("\t Generating Demand knowledge solutions")
        initial_counter = (partitions+rest+CUSTOM_SOLUTIONS+(solutions_for_knowledge))
        total_counter = initial_counter + (solutions_for_knowledge)
        for counter in range(initial_counter, total_counter):
            indv = val[solutions].reshape(1, -1)
            
            try:
                prob_var = probablity_list[solutions-initial_counter]
                print(f"counter={counter}, solutions={solutions}, initial_counter={initial_counter}, len(probablity_list)={len(probablity_list)}", flush=True)
            except:
                print("DEM", flush=True, file=sys.stderr)
                print(f"IndexError: solutions={solutions}, initial_counter={initial_counter}, len(probablity_list)={len(probablity_list)}", flush=True,file=sys.stderr)
                print(f"solutions-initial_counter={solutions-initial_counter}", flush=True, file=sys.stderr)
                raise
            
            mutation = self.mutation_dict.get("DEM")            
            val[solutions] = mutation._do(problem, X=indv, prob_var=prob_var)
            solutions +=1
        # with open("val_after_demand.pkl", "wb") as f:
        #     pickle.dump(val, f)
        # print("\tsolutions{}={}".format(counter,solutions))
        # Multimodal knowledge solutions
        # print("\t Generating Multimodal knowledge solutions")

        initial_counter = initial_counter + (solutions_for_knowledge)
        total_counter = initial_counter + (solutions_for_knowledge)
        for counter in range(initial_counter, total_counter):
            indv = val[solutions].reshape(1, -1)
            try:
                print("MOD", flush=True, file=sys.stderr)
                print(f"solutions-initial_counter={solutions-initial_counter}", flush=True)
                print(f"solutions={solutions}, initial_counter={initial_counter}, solutions-initial_counter={solutions-initial_counter}, len(probablity_list)={len(probablity_list)}", flush=True)
                prob_var = probablity_list[solutions-initial_counter]
            except:
                print("MOD", flush=True)
                print(f"IndexError: solutions={solutions}, initial_counter={initial_counter}, len(probablity_list)={len(probablity_list)}", flush=True)
                print(f"solutions-initial_counter={solutions-initial_counter}", flush=True)
                raise
            mutation = self.mutation_dict.get("MOD")
            val[solutions] = mutation._do(problem, X=indv, prob_var=prob_var)
            solutions +=1

        shape_after = val.shape
        assert shape_before == shape_after, f"Shape before: {shape_before}, Shape after: {shape_after}"
        
        assert solutions == n_samples, f"Expected {n_samples} solutions, but generated {solutions}."
        return val #.reshape(n_samples, problem.n_var).astype(bool)

class OneKnowledgeSampling(BinaryRandomSampling):
    """
    This abstract class represents the initialization sampling
    that creates a initial population considering
    One Knowlege, Random and Custom
    The Custom solutions are provided by the user in a dataframe with the column "individuals" containing the binary representation of the solution.

    n_samples: number of samples to generate
    problem.n_var: number of variables in the problem(Number of states (0,1) in the binary representation of the solution LEN_SEARCH_SPACE)    

    32 solutions will be generated in total, 8 random, 3 custom, 21 one knowledge.

    8 will be generated randomly,
    3 will be generated from the custom solutions provided by the user,
    21 will be generated from the current solution and then applying the connectivity knowledge with a probablity of
        {1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, 1}
    """
    def __init__(self, df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list) -> None:
            super().__init__()
            self.individual_custom = df_individual
            
            # MOD_Mutation = KnowledgeMOD(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment")
            # CON_Mutation = KnowledgeCON(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment",verbose=True)
            # DEM_Mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
            # self.mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
            # self.mutation_op = "DEM"
            self.mutation = None
            self.mutation_op = ""
            print("Knowledge_Created")
            # self.mutation_dict = mutation_dict
            
    def _do(self, problem, n_samples):
        print(f"Generating {n_samples} samples for problem with {problem.n_var} variables.", flush=True)
        val = np.zeros((n_samples, problem.n_var), dtype=bool)
        val_before = val.shape
        partitions = int(n_samples/4)
        rest = n_samples % 4
        CUSTOM_SOLUTIONS = 3

        if rest != 0:
            print(f"Warning: n_samples ({n_samples}) is not divisible by 4. The rest of the division will be added to the random partition.")

        partitions = int(partitions)
        solutions_for_knowledge = int(partitions - 1 )*3# for custom
        probablity_list = [i/(solutions_for_knowledge) for i in range(1, solutions_for_knowledge+1)]  # [1/7, 2/7, ..., 7/7]    

        print(f"len(probablity_list)={len(probablity_list)}")
        solutions = 0
        
        # 1. Random solutions (8)
        # print("\tGenerating RANDOM solutions")
        random_solutions_count = partitions + rest
        for counter in range(0, random_solutions_count):
            val[solutions] = (np.random.random(problem.n_var) < 0.5).astype(bool)
            solutions += 1
        
        # 2. Custom solutions (3)
        # print("\tGenerating CUSTOM knowledge solutions")
        
        init_counter = random_solutions_count
        total_counter = (random_solutions_count+CUSTOM_SOLUTIONS)
        for counter , indv in zip (range(init_counter, total_counter), self.individual_custom.itertuples()):
            new_ind = eval(getattr(indv,"individuals"))
            val[solutions] = new_ind
            counter += 1
            solutions +=1
        
        # 3. Mutation knowledge solutions (21)
        # print(f"\tGenerating {self.mutation_op} knowledge solutions")
        init_counter = (random_solutions_count+CUSTOM_SOLUTIONS)
        total_counter = (partitions+rest+CUSTOM_SOLUTIONS+(solutions_for_knowledge))
        for counter in range(init_counter, total_counter):
            
            prob_var = probablity_list[counter - init_counter]
            try:
                prob_var = probablity_list[counter - init_counter]
            except:
                print("ALL", flush=True)
                print(f"counter: {counter}", flush=True)
                print(f"IndexError: solutions={solutions}, initial_counter={init_counter}, len(probablity_list)={len(probablity_list)}", flush=True)
                print(f"solutions-initial_counter={solutions-init_counter}", flush=True)
                raise            
            print(f"\t\tprob_var={prob_var}, solutions_idx={solutions}")
            indv = val[solutions].reshape(1, -1)  # ← Usa 'solutions', NO 'counter + sol'
            val[solutions] = self.mutation._do(problem, X=indv, prob_var=prob_var)
            solutions += 1
        val_after = val.shape
        print(f"\tN samples = {n_samples}")
        print(f"\tTotal solutions generated: {solutions}")
        assert val_before == val_after, f"Shape before: {val_before}, Shape after: {val_after}"
        assert solutions == n_samples, f"Expected {n_samples} solutions, but generated {solutions}."
        
        return val #.reshape(n_samples, problem.n_var).astype(dtype=bool)

    # def _do(self, problem, n_samples):
    #         # val = np..random((n_samples, problem.n_var))
    #         val = np.zeros((n_samples, problem.n_var), dtype=bool)
    #         probablity_list = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, 1]
    #         # Random solutions
    #         solutions = 0
    #         for counter in range(0, 8):
    #             val[counter] = (np.random.random(problem.n_var) < 0.5).astype(bool)
    #             solutions += 1
    #         # Custom solutions

    #         print("\tGenerating CUSTOM knowledge solutions")
    #         counter = 8
    #         for indv in self.individual_custom.itertuples():
    #             new_ind = eval(getattr(indv,"individuals"))
    #             val[counter] = new_ind
    #             counter += 1
    #             solutions += 1
    #              # Connectivity knowledge solutions
    #         print(f"\tGenerating {self.mutation_op}  knowledge solutions")
    
    #         for sol, prob_var in enumerate(probablity_list):
    #             # prob_var = probablity_list[counter-11]
    #             solutions_of_know = counter + sol
    #             print(f"\t\tprob_var={prob_var},counter={counter}")
    #             indv = val[solutions_of_know].reshape(1, -1)
    #             val[solutions_of_know] =  self.mutation._do(problem, X=indv, prob_var=prob_var)
    #             solutions += 1

    #         print(f"\tTotal solutions generated: {solutions}")
    #         # assert solutions == n_samples, f"Expected {n_samples} solutions, but generated {solutions}."
    #         return val.reshape(n_samples, problem.n_var).astype(dtype=bool)

class ConnectivityKnowledgeSampling(OneKnowledgeSampling):
    """
        This abstract class represents the initialization sampling
        that creates a initial population considering
        Connecitivy, Random and Custom
        The Custom solutions are provided by the user in a dataframe with the column "individuals" containing the binary representation of the solution.

        n_samples: number of samples to generate
        problem.n_var: number of variables in the problem(Number of states (0,1) in the binary representation of the solution LEN_SEARCH_SPACE)    

        32 solutions will be generated in total, 8 random, 3 custom, 21 connectivity.

        8 will be generated randomly,
        3 will be generated from the custom solutions provided by the user,
        21 will be generated from the current solution and then applying the connectivity knowledge with a probablity of
          {1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, 1}
        
    """
    def __init__(self, df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list) -> None:
            super().__init__(df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list)
            self.individual_custom = df_individual
            
            # MOD_Mutation = KnowledgeMOD(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment")
            CON_Mutation = KnowledgeCON(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment",verbose=True)
            # DEM_Mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
            self.mutation = CON_Mutation
            self.mutation_op = "CON"
            print("Knowledge_Created")
            # self.mutation_dict = mutation_dict
            
        



class DemandKnowledgeSampling(OneKnowledgeSampling):
    """
        This abstract class represents the initialization sampling
        that creates a initial population considering
        5 Kind of solutions: Connectivity, Demand, Multimodal, Random and Custom
        The Custom solutions are provided by the user in a dataframe with the column "individuals" containing the binary representation of the solution.

        n_samples: number of samples to generate
        problem.n_var: number of variables in the problem(Number of states (0,1) in the binary representation of the solution LEN_SEARCH_SPACE)    

        32 solutions will be generated in total, 8 random, 3 custom, 21 demand knowledge.

        8 will be generated randomly,
        3 will be generated from the custom solutions provided by the user,
        21 will be generated from the current solution and then applying the demand knowledge with a probablity of
          {1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, 1}
        
    """
    def __init__(self, df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list) -> None:
        super().__init__(df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list)
        self.individual_custom = df_individual
        
        # MOD_Mutation = KnowledgeMOD(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment")
        # CON_Mutation = KnowledgeCON(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment",verbose=True)
        DEM_Mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
        self.mutation = DEM_Mutation
        self.mutation_op = "DEM"
        print("Knowledge_Created")
        # self.mutation_dict = mutation_dict


class MultimodalKnowledgeSampling(OneKnowledgeSampling):
    """
        This abstract class represents the initialization sampling
        that creates a initial population considering
        5 Kind of solutions: Connectivity, Demand, Multimodal, Random and Custom
        The Custom solutions are provided by the user in a dataframe with the column "individuals" containing the binary representation of the solution.

        n_samples: number of samples to generate
        problem.n_var: number of variables in the problem(Number of states (0,1) in the binary representation of the solution LEN_SEARCH_SPACE)    

        32 solutions will be generated in total, 8 random, 3 custom, 21 multimodal.

        8 will be generated randomly,
        3 will be generated from the custom solutions provided by the user,
        21 will be generated from the current solution and then applying the multimodal knowledge with a probablity of
          {1/21, 2/21, 3/21, 4/21, 5/21, 6/21, 7/21, 8/21, 9/21, 10/21, 11/21, 12/21, 13/21, 14/21, 15/21, 16/21, 17/21, 18/21, 19/21, 20/21, 1}
        
    """
    def __init__(self, df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list) -> None:
        super().__init__(df_individual, G, dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE,weight_metric, pair_list)
        self.individual_custom = df_individual
        
        MOD_Mutation = KnowledgeMOD(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment")
        # CON_Mutation = KnowledgeCON(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,prob_var=0,weight_metric="time_experiment",verbose=True)
        # DEM_Mutation = KnowledgeDEM(graph=G, dict_of_search_space_u_v=dict_of_search_space_u_v,dict_of_search_space_uvk_to_pos=dict_of_search_space_uvk_to_pos,LEN_SEARCH_SPACE=LEN_SEARCH_SPACE,weight_metric=weight_metric, pair_list=pair_list,prob_var=1)
        self.mutation = MOD_Mutation
        self.mutation_op = "MOD"
        print("Knowledge_Created")
        # self.mutation_dict = mutation_dict


