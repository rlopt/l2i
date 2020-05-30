# A Learning-based Iterative Method for Solving Vehicle Routing Problems
A learning-based algorithm for solving the Travelling Salesman Problem (TSP) and the Vehicle Routing Problem (VRP).
# Paper
For more details, please see our paper [A Learning-based Iterative Method for Solving Vehicle Routing Problems](https://openreview.net/pdf?id=BJe1334YDH), which has been accepted at ICLR 2020.
# Dependencies
* Python>=3.6
* TensorFlow
# Quick Start
For training CVRP-20 with 2000 problem instances:
```python
python ml_opt.py --num_training_points 20 --num_test_points 20 --num_episode 2000
```
