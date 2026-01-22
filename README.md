# Executable Science and Applications Traceability Matrix for Earth-observation Missions 

ExecSATM is a Python library for representing, evaluating, and executing a Science and Applications Traceability Matrix (SATM) for autonomous Earth-observation missions.

It provides a formal, executable mapping between mission objectives, measurement requirements, observation tasks, and satellite-specific execution costs, enabling quantitative valuation and prioritization of observations in both nominal and event-driven operations.

ExecSATM supports multi-mission-objective science value modeling, attribute-based preference functions, task relevance mapping, and utility-based task evaluation for distributed and autonomous satellite constellations.

## Installation
Requires: Unix-like operating system (Linux (Ubuntu, CentOS...), Mac), `python >=3.8`, `pip`, `make`

The installation can be carried out in a conda environment using the below steps.

`pip install setuptools` is required for Python 3.12 for the `distutils` package.

1. Create and activate a new conda environment with python. Install `pip` in the environment.

```
conda create --name foo python=3.8
conda activate foo
conda install pip
```

2. Install `execsatm` library.
```
make 
```

2. **Optional:** Confirm installation.
```
make runtest
```

## Acknowledgments
This work was supported by the National Aeronautics and Space Administration (NASA) Earth Science Technology Office (ESTO) through the Advanced Information Systems Technology (AIST) Program, and by the Mexican Ministry of Science, Humanities, Technology, and Innovation (SECIHTI) through its Graduate Scholarships for Studies in Science and Humanities Abroad Fellowship.

## Contact 
**Principal Investigator:** 
- Daniel Selva Valero - <dselva@tamu.edu>

**Lead Developers:** 
- Alan Aguilar Jaramillo - <aguilaraj15@tamu.edu>
