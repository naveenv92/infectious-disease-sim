# Infectious Disease Simulation

Basic set of classes to try and capture the behavior of the propagation of an infectious disease through a population.

## Files  
simulation.py – collection of classes for running simulation  
sampleSimulation.py – script to use classes from `simulation` to run a simulation  

## Classes  
### Particle  
Constructor  
```python
Particle(x, y, vx, vy, r)
```
`x` -- x-coordinate of particle (default 0)  
`y` -- y-coordinate of particle (default 0)  
`vx` -- x-component of particle velocity (default 1)  
`vy` -- y-component of particle velocity (default 1)  
`r` -- size of particle for viewing simulation (default 2)  