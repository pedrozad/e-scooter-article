# Infrastructure-Redesigining

## Requirements
Python version 3.10.12 64 bits
and packages in environment.yml

## To ejecute the project
```bash
conda env create -f environment.yml

python multiobjective_scooter.py -seed [SEED] -pc [PROB_CROSS] -pm [PROB_FLIP] -POB [POPULTATION] -GEN [Number of Generations]  -CPUS [CPUS] -a [ALGORITHM] -f [file_initialization_especial_individuals]  -m [Map_of_city] -mut [Mutation_type]
```
or 

```bash
conda env create -f environment.yml

python multiobjective_scooter.py @arguments.txt
```


# Other arguments
-show_ind -> Show the individuals

-v -> Show the individuals

-type_reference=[reference direction]

-np=[points or partitions depending of the reference direction: FLOAT]

-prob_neighbor_mating=[FLOAT]

-n_neighbors=[NUMBER]




## Structure of the project
Main
* multiobjective_scooter.py
* mutation.py
* sampling.py
* data-osm
    * Malaga-Subway
		* districts-Malaga-Subway-data-with-nodes.csv
		* over_all_base_withBus.csv
		* pair_less_than_3600_new_points.csv
		* maps
			* map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY-wBUS-wMetro.gpkg
	* Melilla
		* districts-Melilla-data-with-nodes.csv
		* over_all_base_withBus.csv
		* pair_less_than_3600_new_points.csv
		* maps
			* map-Melilla-all--scooter-walking-bus.gpkg
* environment.yml
* arguments.txt

## Data

The data folder are in https://uma365-my.sharepoint.com/:f:/g/personal/pedroza_uma_es/EveivO1mkrZAhhc7Dqu7k_oBg0Kl-a9_M1fkasqlM9eHew?e=DxSoog

## Arguments
Example of arguments can be found in arguments.txt
