# Infrastructure Redesigning

Source code for Multiobjective Optimization of E-scooter Infrastructure using Evolutionary Algorithms.

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
  -s <SEED> \
  -pc <PROB_CROSS> \
  -pm <PROB_FLIP> \
  -POB <POPULATION> \
  -GEN <NUM_GENERATIONS> \
  -CPUS <NUM_CPUS> \
  -a <ALGORITHM> \
  -f <FILE_SPECIAL_INDIVIDUALS> \
  -m <SCENARIO> \
  -mut <MUTATION_TYPE> \
  -sampling <INITIALIZATION_TYPE> \
  -suffix <RESULTS_SUFFIX>
```

### Using an arguments file

Create an `arguments.txt` file with one argument per line (see `arguments.txt` for an example), then run:

```bash
apptainer exec e-scooter.sif python multiobjective_scooter_scenario.py @arguments.txt
```

### Using Conda

```bash
python multiobjective_scooter_scenario.py \
  -s <SEED> \
  -pc <PROB_CROSS> \
  -pm <PROB_FLIP> \
  -POB <POPULATION> \
  -GEN <NUM_GENERATIONS> \
  -CPUS <NUM_CPUS> \
  -a <ALGORITHM> \
  -f <FILE_SPECIAL_INDIVIDUALS> \
  -m <SCENARIO> \
  -mut <MUTATION_TYPE> \
  -sampling <INITIALIZATION_TYPE> \
  -suffix <RESULTS_SUFFIX>
```

## Arguments

| Argument | Description | Type | Valid values |
|---|---|---|---|
| `-s` | Random seed for reproducibility | INT | Any integer |
| `-pc` | Crossover probability | FLOAT | [0, 1] |
| `-pm` | Mutation probability for each bit | FLOAT | [0, 1] |
| `-POB` | Population size | INT | Positive integer |
| `-GEN` | Number of generations | INT | Positive integer |
| `-CPUS` | Number of worker processes | INT | Positive integer |
| `-a` | Algorithm to run | STRING | `NSGA-II`, `NSGA-II`|
| `-f` | File with initial individuals | PATH | CSV file path or omitted |
| `-m` | Scenario and input data set | STRING | `Malaga`, `Melilla` |
| `-mut` | Mutation type | STRING | `MBF`, `DEM`, `CON`, `MOD`, `AMR_M`, `AMR_D`, `AMR_C` |
| `-sampling` | Initialization type | STRING | `base_init`, `con_knowledge`, `mod_knowledge`, `dem_knowledge` |
| `-suffix` | Suffix for the results directory | STRING | Any string or omitted |

### Additional arguments

| Argument | Description | Type | Valid values |
|---|---|---|---|
| `-show_ind` | Show individuals | FLAG | Present or absent |
| `-v` | Verbose output / show individuals | FLAG | Present or absent |
| `-type_reference` | Reference direction type | STRING | `das-dennis`, `energy` |
| `-np` | Number of points or partitions, depending on the reference direction | INT | Positive integer |
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

[Download data folder](https://uma365-my.sharepoint.com/:f:/g/personal/pedroza_uma_es/IgB8ItAJ4ghNSo1taveje00bAZiUAEZ0Kic77_HkTH4Mn3Y?e=dOQEJ2)

## Example

Example argument configurations can be found in `arguments.txt`.

## Expected Output

After a successful run, results are saved under `results/<SCENARIO>/`, with an
additional subfolder when `-suffix` is provided. This folder contains the
Pareto front and the fitness values of the population in the last generation.
If `-v` is present, the record for each generation will also be available. If
`-show_ind` is present, generation output files will include the individuals.

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

- Code: MIT License
- Map data: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright)
- Other data: MIT License


