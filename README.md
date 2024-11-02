# MD_code_test

The aim of the task is to program a basic molecular dynamics (MD) program and present this at an interview.

# Google colab notebook

See:
https://colab.research.google.com/drive/1vNaAClYGbcYyL-Aielj3nE7nthR0kt1J?usp=sharing

# Molecular Dynamics (MD) Program Task (see pdf of original)

## Objective

Create a basic molecular dynamics program using classical mechanics (no quantum mechanics or relativity theory). The program should numerically integrate Newton's equations of motion for a finite number of time steps.

## System Description

- **Particles**: N particles with mass m at temperature T
- **Dimension**: 2D simulation (x,y coordinates)
- **Interactions**: Pairwise van der Waals interactions (Lennard-Jones potential)
- **Simulation Box**: Square box with side length 20 Å (Angstroms)
- **Boundary Conditions**: Reflective boundaries at x₁=0, x_f=20 Å, y₁=0, y_f=20 Å

### Lennard-Jones Potential

The interaction between particles is modeled using the Lennard-Jones potential, where:

- r_ij is the distance between particles i and j
- σ is the particle size parameter
- ε is the interaction strength parameter

## Simulation Parameters

For simulating 16 argon atoms (N=16):

- **Mass (m)**: 39.948 g/mol
- **Temperature (T)**: 300 K
- **Particle Size (σ)**: 3.4 Å
- **Interaction Strength (ε)**: 0.24 kcal/mol

## Simplifications

Note that the following simplifications are applied:

- Two-dimensional system
- No temperature/pressure control
- Non-periodic boundary conditions

Due to these simplifications, exact argon gas properties may not be reproduced.

## Implementation Steps

### 1. Initial Setup

- Read initial coordinates and velocities from `Ar_initial.txt`
- File format: Each section contains columns for atom ID, x component, and y component

### 2. Force Calculation

- Compute forces on each atom due to all other atoms
- Use the derivative of the Lennard-Jones potential
- Print initial forces for the given coordinates

### 3. Integration

- Implement an integrator for Newton's equations of motion
- Use appropriate time step Δt
- Perform minimum 10,000 integrations
- Recommended: Use velocity-Verlet algorithm

#### Suggestion

Create a function that:

- Input: current positions and velocities at time step n
- Output: positions and velocities at time step n+1

### 4. Energy Analysis

Every 100 steps, calculate and plot vs. time:

- Potential energy
- Kinetic energy
- Instantaneous temperature
- Total energy (potential + kinetic)
  All energies should be expressed per mol.

### 5. Trajectory Visualization

- Plot particle positions every 100 steps
- Visualize argon atom motion within the box

## Note on Reflective Boundary Conditions

When a particle crosses a boundary:

1. Flip the sign of the velocity component orthogonal to the boundary
2. Adjust position as if particle underwent elastic collision

Example:

- Initial state: x=0.1, vx=-0.3
- Boundary at x=0
- After time step (Δt=1): x would be -0.2
- Correction:
  - New x-position: 0.2
  - New x-velocity: +0.3 (reversed)

This implements elastic collision mechanics at boundaries.
