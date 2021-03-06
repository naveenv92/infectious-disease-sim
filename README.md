# Infectious Disease Simulation

Basic set of classes to try and capture the behavior of the propagation of an infectious disease through a population.  
&nbsp;  
&nbsp;  

## Files  
`simulation.py` &ndash; collection of classes for running simulation  
`sampleSimulation.py` &ndash; sample script to use classes from `simulation` to run a simulation  
`sampleSimulation.ipynb` &ndash; sample Jupyter notebook (functions same as `sampleSimulation.py`)
&nbsp;  
&nbsp;  

## `class Particle` 
```python
Particle(x=0, y=0, vx=1, vy=1, r=5)
```
`x` &ndash; x-coordinate of particle (default `0`)  
`y` &ndash; y-coordinate of particle (default `0`)  
`vx` &ndash; x-component of particle velocity (default `1`)  
`vy` &ndash; y-component of particle velocity (default `1`)  
`r` &ndash; size of particle for viewing simulation (default `5`)  

```python
Particle.x()
```
Returns the x-coordinate of the particle  

```python
Particle.set_x(x)
```
Set x-coordinate of the particle to `x`  

```python
Particle.y()
```
Returns y-coordinate of the particle  

```python
Particle.set_y(y)
```
Set y-coordinate of the particle to `y`  

```python
Particle.vx()
```
Returns the x-component of the velocity  

```python
Particle.set_vx(vx)
```
Set x-component of the velocity to `vx`  

```python
Particle.vy()
```
Returns the y-component of the velocity  

```python
Particle.set_vy(vy)
```
Set the y-component of the velocity to `vy`  

```python
Particle.r()
```
Returns the size of the particle for viewing the simulation  

```python
Particle.set_r(r)
```
Set the size of particle to `r`  

```python
Particle.move(dt)
```
Move the particle by a timestep determined by `dt` (default is `1`), and if infected, countdown until recovery  

```python
Particle.infect()
```
Infects the particle if it has not been previously infected or has recovered from infection  

```python
Particle.isInfected()
```
Returns whether the particle is infected (boolean)  

```python
Particle.isRecovered()
```
Returns whether the particle has recovered (boolean)  

```python
Particle.infectOtherParticle()
```
Increment the number of particles infected by the current particle  

```python
Particle.numOfInfections()
```
Returns the number of particles infected by the current particle  
&nbsp;  
&nbsp;  

## `class Simulation`
```python
Simulation(n=100, ninf=1, r=5, boxSize=1, speed=1, tol=0.1)
```
`n` &ndash; number of particles in the simulation box (default `100`)  
`ninf` &ndash; number of initially infected particles (default `1`)  
`r` &ndash; size of particle for viewing simulation (default `5`)  
`boxSize` &ndash; size of simulation box in each direction <i>i.e.</i> 1 &times; 1, 2 &times; 2, 3 &times; 3, etc. (default `1`)  
`speed` &ndash; maximum value of each velocity component, magnitude of speed is bounded by [0, &radic;2 &middot; `speed`] (default `1`)  
`tol` &ndash; amount of tolerance around each particle used to determine collisions (default `0.1`)  

```python
Simulation.n()
```
Returns the number of particles in the simulation  

```python
Simulation.getBoxSize()
```
Returns the size of the simulation box  

```python
Simulation.coords()
```
Returns arrays of x and y coordinates of non-infected particles  

```python
Simulation.coords_inf()
```
Returns array of x and y coordinates of infected particles  

```python
Simulation.radii()
```
Returns size of particles for viewing simulation  

```python
Simulation.move()
```
Move each particle in the simulation and manage collisions  

```python
Simulation.statistics()
```
Returns a list of the form `[countNI, countInf, countRec, avgInfections]`:  
`countNI` &ndash; number of non-infected particles at each timestep  
`countInf` &ndash; number of infected particles at each timestep  
`countRec` &ndash; number of recovered particles at each timestep  
`avgInfections` &ndash; average number of particles infected by each infected particle (~R<sub>0</sub> value)  
&nbsp;  
&nbsp;  

## `class runSimulation`
```python
runSimulation(simulation)
```
`simulation` &ndash; `Simulation` object used to initialize class

```python
runSimulation.singleSim(showSim=True, plot=True)
```
Runs a single simulation and saves the resulting statistics as a `.csv`  file  
`showSim` &ndash; boolean input for whether to show the simulation window (default `True`)  
`plot` &ndash; boolean input for whether to plot final statistics (default `True`)  