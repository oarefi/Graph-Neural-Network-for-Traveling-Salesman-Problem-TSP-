# Graph Neural Network for Traveling Salesman Problem (TSP)
This repository contains code for solving the Traveling Salesman Problem (TSP) using a Graph Neural Network (GNN). The TSP is a classic combinatorial optimization problem where the goal is to find the shortest possible route that visits a set of cities and returns to the starting city.

## Requirements
To run the code, you need the following dependencies:

## Python 3.x
* PyTorch (with Torch Geometric)
* OR-Tools
* Matplotlib
* NumPy

You can install the required dependencies using the following commands:
'''
pip install torch torch-geometric ortools matplotlib numpy
'''
## TSP_GNN Model
The TSP_GNN model is a graph neural network designed to solve the TSP. It takes as input the node positions (2D coordinates) of cities and uses GAT (Graph Attention) and GCN (Graph Convolutional Network) layers for message passing and aggregation. The model then outputs node embeddings that are used to determine the TSP tour.

## How to Use
You can use the code to run experiments for solving the TSP using the GNN model and compare the results with optimal solutions found using OR-Tools.

Import the required libraries and define the TSP_GNN model.

Define functions to calculate the total distance traveled in a tour and to visualize the TSP tour and city locations.

Implement the function solve_tsp_optimal using OR-Tools to solve TSP optimally.

Implement the function run_experiment to perform multiple runs of the TSP problem and compare GNN's performance with optimal solutions.

Set the parameters for the experiment, such as the number of nodes and the GNN hidden dimension.

Run the experiment using the run_experiment function.

The code will run multiple experiments, visualize the TSP tour found by GNN, and compare it with the optimal solution. It will also print the total distance traveled by each tour and the number of times GNN outperforms the optimal tour.

## Results
The GNN model demonstrates good performance in finding approximate solutions to the TSP. The comparison of the total distances traveled by GNN and optimal tours provides insights into the model's accuracy. The average approximation ratio and execution times for GNN and OR-Tools are also displayed.

## Conclusion
This code demonstrates how to use a Graph Neural Network to solve the Traveling Salesman Problem. By running experiments and comparing the results with optimal solutions, you can assess the GNN's performance in solving this challenging optimization problem. The GNN's ability to approximate TSP tours efficiently makes it a promising approach for other combinatorial optimization tasks as well.
