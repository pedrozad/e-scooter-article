# Infrastructure Redesigning

Source code for multi-objective optimization of e-scooter infrastructure using evolutionary algorithms.

## Requirements

- Python 3.10.12 (64-bit)
- [Apptainer](https://apptainer.org/) is recommended for reproducible execution
- Alternatively, Conda can be used with the provided environment file

## Installation

### Option 1: Apptainer (recommended)

Build the container image:

```bash
apptainer build e-scooter.sif e-scooter.def
```

### Option 2: Conda

Create and activate the environment:

```bash
conda env create -f environment_cluster.yml
conda activate <environment_name>
```

## Running the Project

### Using Apptainer

```bash
apptainer exec e-scooter.sif python multiobjective_scooter_scenario.py \
  -seed <SEED> \
  -pc <PROB_CROSS> \
  -pm <PROB_FLIP> \
  -POB <POPULATION> \
  -GEN <NUM_GENERATIONS> \
  -CPUS <NUM_CPUS> \
  -a <ALGORITHM> \
  -f <FILE_SPECIAL_INDIVIDUALS> \
  -m <CITY_MAP> \
  -mut <MUTATION_TYPE> \
  -sampling <INITIALIZATION_TYPE> \
  -sif <SCENARIO_NAME>
```

### Using an arguments file

Create an `arguments.txt` file with one argument per line (see `arguments.txt` for an example), then run:

```bash
apptainer exec e-scooter.sif python multiobjective_scooter_scenario.py @arguments.txt
```

### Using Conda

```bash
python multiobjective_scooter_scenario.py \
  -seed <SEED> \
  -pc <PROB_CROSS> \
  -pm <PROB_FLIP> \
  -POB <POPULATION> \
  -GEN <NUM_GENERATIONS> \
  -CPUS <NUM_CPUS> \
  -a <ALGORITHM> \
  -f <FILE_SPECIAL_INDIVIDUALS> \
  -m <CITY_MAP> \
  -mut <MUTATION_TYPE> \
  -sampling <INITIALIZATION_TYPE> \
  -sif <SCENARIO_NAME>
```

## Arguments

| Argument | Description | Type | Valid values |
|---|---|---|---|
| `-seed` | Random seed for reproducibility | INT | Any integer |
| `-pc` | Crossover probability | FLOAT | [0, 1] |
| `-pm` | Mutation probability | FLOAT | [0, 1] |
| `-POB` | Population size | INT | Any positive integer |
| `-GEN` | Number of generations | INT | Any positive integer |
| `-CPUS` | Number of CPUs to use | INT | Any positive integer |
| `-a` | Algorithm to run | STRING | `NSGA-II`, `NSGA-III` |
| `-f` | File with special individuals for initialization | PATH | CSV file path |
| `-m` | City map file | PATH | `.gpkg` file path |
| `-mut` | Mutation type | STRING | e.g., `polynomial`, `bitflip` |
| `-sampling` | Initialization type | STRING | e.g., `random`, `latin_hypercube` |
| `-sif` | Scenario name, used to create a control subfolder | STRING | Any string |

### Additional arguments

| Argument | Description | Type | Valid values |
|---|---|---|---|
| `-show_ind` | Show individuals | FLAG | Present or absent |
| `-v` | Verbose output / show individuals | FLAG | Present or absent |
| `-type_reference` | Reference direction type | STRING | e.g., `uniform`, `das_dennis` |
| `-np` | Number of points or partitions, depending on the reference direction | FLOAT | Positive number |
| `-prob_neighbor_mating` | Probability of neighbor mating | FLOAT | [0, 1] |
| `-n_neighbors` | Number of neighbors | INT | Positive integer |

## Project Structure

```
.
├── multiobjective_scooter_scenario.py
├── mutation.py
├── sampling.py
├── environment.yml
├── arguments.txt
├── data-osm/
│   └── Malaga-Subway/
├── districts-Malaga-Subway-data-with-nodes.csv
├── over_all_base_withBus.csv
├── pair_less_than_3600_new_points.csv
├── maps/
│   └── map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY-wBUS-wMetro.gpkg
├── Melilla/
│   ├── districts-Melilla-data-with-nodes.csv
│   ├── over_all_base_withBus.csv
│   ├── pair_less_than_3600_new_points.csv
│   └── maps/
│       └── map-Melilla-all--scooter-walking-bus.gpkg
```

## Data

Input data files are available at:

[Download data folder](https://uma365-my.sharepoint.com/:f:/g/personal/pedroza_uma_es/EveivO1mkrZAhhc7Dqu7k_oBg0Kl-a9_M1fkasqlM9eHew?e=DxSoog)

## Example

Example argument configurations can be found in `arguments.txt`.

## Expected Output

After a successful run, the results are saved in a subfolder named after the `-sif` scenario argument. This folder contains the Pareto front and the fitness values of the population in the last generation. If `-v` is present, the record for each generation will also be available. If `-show_ind` is present, all output files will include the individuals, except for the Pareto front.

Execution logs are written to `stdout` and `stderr`, respectively.


## Citation

If you use this code or data in your research, please cite the associated paper:

```bibtex
@article{Pedroza-Perez2026,
  title   = {Improving Micromobility Integration through Problem-Specific Evolutionary Operators: A Multiobjective Framework for Urban Road Redesign},
  author  = {Pedroza-Perez, Diego Daniel and Toutouh, Jamal and Luque, Gabriel},
  journal = {Applied Soft Computing},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {}
}
```


## License

This project is licensed under the [MIT License](LICENSE).


