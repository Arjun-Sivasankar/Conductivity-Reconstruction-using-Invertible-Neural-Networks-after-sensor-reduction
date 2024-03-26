# Conductivity-Reconstruction-using-Invertible-Neural-Networks-after-sensor-reduction
Research Project : M1107-CMS05

## Overview
The transition to sustainable energy systems hinges on advancements in electrolysis technology for hydrogen production, where efficiency is critical. Gas bubbles, which inhibit  electrolytic reactions and diminish overall efficiency, are a major stumbling block in this process. This project tackles the problem of bubble identification and dispersion in  electrolyzers by using Invertible Neural Networks (INNs) to create high-resolution conductivity maps from low-resolution magnetic field measurements. We initially investigated a dataset
of 10,000 configurations, each consisting of 131,072 sensor features to attain 5600 current values for 2D current distribution reconstruction, but encountered problems due to data
sparsity and redundancy. As a result, we took a more data-efficient strategy, using 10,000 configurations of 100 sensor data points each to achieve 510 values of conductivity
for each configuration giving rise to reconstructed conductivity maps. Our methodology involves deploying INNs with an emphasis on sensor reduction, with the goal of maintaining
reconstruction accuracy while minimizing the sensor array. A baseline model with an average loss of 0.828 was used as the performance benchmark. Following that, we used a variety of
sensor reduction strategies, including random sampling, naive grid reduction, and algorithmic approaches such as PCA with selection, simulated annealing, and genetic algorithms, to
identify optimal sensor configurations that approximated the baseline model’s accuracy. The project’s outcome suggests that intelligent sensor selection can significantly reduce sensor
count while preserving reconstruction quality, thereby contributing to the development of more efficient electrolysis systems for clean energy production.

## Installation

### Prerequisites
- Python 
- PyTorch 


### Setup
Clone the repository to your local machine:

Usage
Clone this repository to your local machine:
```bash
  git clone https://github.com/Arjun-Sivasankar/Conductivity-Reconstruction-using-Invertible-Neural-Networks-after-sensor-reduction.git
```