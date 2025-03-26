# 🏭 Flow Shop Scheduling Problem (FSSP) Solver

A platform for solving Flow Shop Scheduling Problems using various heuristic optimization algorithms.

## 📋 Description

This repository contains a comprehensive platform for solving the Flow Shop Scheduling Problem (FSSP), a well-known optimization problem in operations research. The platform implements multiple metaheuristic algorithms to find near-optimal solutions for scheduling jobs across machines while minimizing the makespan (total completion time).

The Flow Shop Scheduling Problem involves scheduling a set of jobs to be processed on a set of machines. Each job must be processed on all machines in the same order, and the goal is to find the sequence of jobs that minimizes the total completion time.

## ✨ Features

- **Multiple Optimization Algorithms**:
  - 🧬 Genetic Algorithm (GA)
  - 🔥 Simulated Annealing (SA)
  - 🐦 Particle Swarm Optimization (PSO)

- **Benchmark Problems**:
  - Includes well-known Taillard benchmark instances
  - Supports different problem sizes (20x5, 20x10, 50x10, 100x10)

- **Visualization Tools**:
  - Gantt charts for visualizing schedules
  - Fitness trend plots for tracking optimization progress
  - Comparative analysis of different algorithms

- **Configurable Platform**:
  - YAML-based configuration for problems, optimizers, and benchmarks
  - Adjustable computational budget
  - Multiple runs for statistical analysis

## 🛠️ Setup Guide

### Prerequisites

- Python 3.6+
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - PyYAML

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flow-shop-scheduling-problem.git
   cd flow-shop-scheduling-problem
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn pyyaml
   ```

## 🚀 Usage

1. Configure the platform by editing the YAML files in the `config/` directory:
   - `general.yaml`: General settings like number of runs and computational budget
   - `problems.yaml`: Problem types to solve
   - `optimizers.yaml`: Optimization algorithms to use
   - `benchmarks.yaml`: Benchmark instances to solve

2. Run the platform:
   ```bash
   python main.py
   ```

3. Results will be displayed as:
   - Gantt charts showing the optimal schedule
   - Fitness trend plots showing the convergence of algorithms
   - Summary statistics in the console and log files

## 📊 Example

```python
# Run the platform with default settings
from main import HeuristicOptimizerPlatform
hop = HeuristicOptimizerPlatform()
```

This will:
1. Load the configured problems, benchmarks, and optimizers
2. Run each enabled optimizer on each enabled benchmark instance
3. Generate visualizations and statistics for the results

## 🧩 Architecture

The repository is organized as follows:

- `main.py`: Main entry point and platform implementation
- `fsspSolver.py`: Alternative solver implementations
- `problems/`: Problem definitions
  - `fssp.py`: Flow Shop Scheduling Problem implementation
  - `jobs.py`: Job representation
  - `machines.py`: Machine representation
  - `problem.py`: Base problem class
- `optimizers/`: Optimization algorithms
  - `ga.py`: Genetic Algorithm
  - `sa.py`: Simulated Annealing
  - `pso.py`: Particle Swarm Optimization
  - `optimizer.py`: Base optimizer class
  - `particle.py`: Particle representation for PSO
- `config/`: Configuration files
  - `general.yaml`: General settings
  - `problems.yaml`: Problem configurations
  - `optimizers.yaml`: Optimizer configurations
  - `benchmarks.yaml`: Benchmark configurations
  - `config.py`: Configuration loader
- `benchmarks/`: Benchmark instances
  - `fssp/`: Flow Shop Scheduling Problem instances
- `utils/`: Utility functions
  - `logger.py`: Logging utilities
  - `stats.py`: Statistical analysis
  - `visualisation.py`: Visualization tools

## 📚 Resources

- [Taillard's FSSP Benchmarks](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html)
- [Flow Shop Scheduling Problem on Wikipedia](https://en.wikipedia.org/wiki/Flow_shop_scheduling)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
