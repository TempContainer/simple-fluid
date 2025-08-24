# Simple-Fluid

A simple fluid simulation using the Warp library. The simulation is based on the Navier-Stokes equations and uses a semi-Lagrangian advection scheme.

## 2D Smoke Simulation

### Overview

- MAC grid
- RK3 time integration
- bicubic interpolation
- semi-Lagrangian scheme
- Poisson solver to enforce incompressibility

![2D Smoke Simulation](out/smoke_2d.gif)

### Requirements

- warp
- matplotlib

## Usage

The repo uses [uv](https://github.com/astral-sh/uv) as package manager. Please refer to the [documentation](https://github.com/astral-sh/uv) for usage instructions.