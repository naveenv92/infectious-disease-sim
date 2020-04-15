from typing import List, Tuple
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


class Particle:
    """
    The Particle object stores position, velocity, and infection status.

    Attributes:
        x (float): x-position of the particle
        y (float): y-position of the particle
        vx (float): x-component of particle velocity
        vy (float): y-component of particle velocity
        recovery_time (int): number of timesteps before recovery
        is_infected (bool): true when particle is infected
        is_recovered (bool): true when particle has recovered

    Methods:

    """

    def __init__(self, x: float, y: float, vx: float, vy: float, recovery_time: int) -> None:
        """
        Parameters:
            x (float): x-position of particle
            y (float): y-position of particle
            vx (float): x-component of particle velocity
            vy (float): y-component of particle velocity
            recovery_time (int): number of timesteps before recovery
        """

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.recovery_time = recovery_time
        self.infected_counter = -1
        self.is_infected = False
        self.is_recovered = False

    def __str__(self) -> str:
        """Return string of Particle object to print."""
        return 'Particle x:%.2f y:%.2f vx:%.2f vy:%.2f' % (self.x, self.y, self.vx, self.vy)

    def __repr__(self) -> str:
        """Return string representation of Particle object."""
        return '<Particle object x:%.2f y:%.2f>' % (self.x, self.y)

    def move(self, dt: float) -> None:
        """
        Moves the particle to its new position after a specified timestep.

        Parameters:
            dt (float): length of timestep
        """

        self.x += self.vx*dt
        self.y += self.vy*dt
        
        if self.is_infected:
            if self.infected_counter == 0:
                self.is_infected = False
                self.is_recovered = True
            elif self.infected_counter > 0:
                self.infected_counter -= 1

    def infect(self) -> None:
        """Infects the particle if it has not already recovered."""

        if not self.is_infected and not self.is_recovered:
            self.is_infected = True
            self.infected_counter = self.recovery_time


class Simulation:
    """
    The Simulation object stores a collection of particles for simulation

    Attributes:

    Methods:
    """

    def __init__(self, n: int, ninf: int, box_size: float, speed: float, recovery_time: int) -> None:
        """
        Parameters:
            n (int): number of particles in the simulation
            ninf (inf): number of initially infected particles
            box_size (float): length of side of simulation box (area = box_size**2)
            speed (float): maximum speed of particles
            recovery_time (int): number of timesteps before recovery
        """

        self.box_size = box_size
        self.infections = ninf
        self.particles = []
        # Healthy particles
        for _ in range(n - ninf):
            x = box_size*np.random.random()
            y = box_size*np.random.random()
            vx = speed*np.random.normal(scale=0.25)
            vy = speed*np.random.normal(scale=0.25)
            self.particles.append(Particle(x, y, vx, vy, recovery_time))
        # Infected particles
        for _ in range(ninf):
            x = box_size*np.random.random()
            y = box_size*np.random.random()
            vx = speed*np.random.normal(scale=0.25)
            vy = speed*np.random.normal(scale=0.25)
            particle = Particle(x, y, vx, vy, recovery_time)
            particle.is_infected = True
            particle.infected_counter = recovery_time
            self.particles.append(particle)

    def __str__(self) -> str:
        """Returns string representation of Simulation object"""
        return 'Simulation n:%d' % (len(self.particles))

    def _handle_collisions(self, particle: Particle) -> None:
        """
        Handles collisions between particles and the walls of simulation box.
        
        If x or y coordinates are outside the box, they are set to 0 and that velocity
        component is reversed. Collisions between particles are currently only used 
        to infect other healthy particles.
        """

        if particle.x < 0:
            particle.x = 0
            particle.vx *= -1
        if particle.x > self.box_size:
            particle.x = self.box_size
            particle.vx *= -1
        if particle.y < 0:
            particle.y = 0
            particle.vy *= -1
        if particle.y > self.box_size:
            particle.y = self.box_size
            particle.vy *= -1

        for other_particle in self.particles:
            if particle == other_particle:
                continue
            elif np.sqrt((particle.x - other_particle.x)**2 + (particle.y - other_particle.y)**2) <= 0.1:
                if particle.is_infected:
                    other_particle.infect()

    def move(self, dt: float) -> None:
        """
        Moves the simulation by one timestep.

        Parameters:
            dt (float): length of the timestep
        """

        num_infections = 0
        for particle in self.particles:
            particle.move(dt)
            self._handle_collisions(particle)
            if particle.is_infected:
                num_infections += 1
        self.infections = num_infections

    def run(self, dt: float) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Runs simulation until no particles are infected

        Parameters:
            dt (float): length of the timestep

        Returns:
            x_healthy (List[List[float]]): x-coordinates of healthy particles at each timestep
            y_healthy (List[List[float]]): y-coordinates of healthy particles at each timestep
            x_inf (List[List[[float]]): x-coordinates of infected particles at each timestep
            y_inf (List[List[[float]]): y-coordinates of infected particles at each timestep
        """

        x_healthy = []
        y_healthy = []
        x_inf = []
        y_inf = []

        while self.infections > 0:
            x_healthy.append([])
            y_healthy.append([])
            x_inf.append([])
            y_inf.append([])
            self.move(dt)
            for particle in self.particles:
                if particle.is_infected:
                    x_inf[-1].append(particle.x)
                    y_inf[-1].append(particle.y)  
                else:
                    x_healthy[-1].append(particle.x)
                    y_healthy[-1].append(particle.y)

        return x_healthy, y_healthy, x_inf, y_inf


def show_simulation(x_healthy: List[float], y_healthy: List[float], x_inf: List[float], y_inf: List[float]):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    healthy, = ax.plot([], [], 'o', color='blue')
    inf, = ax.plot([], [], 'o', color='black')

    def animate(i: int):
        healthy.set_data(x_healthy[i], y_healthy[i])
        inf.set_data(x_inf[i], y_inf[i])
        return healthy, inf

    ani = FuncAnimation(fig, animate, frames=range(len(x_healthy)), interval=100, repeat=False)
    #ani.save('test.mp4')
    plt.show()




if __name__ == "__main__":
    print('Generating simulation with 30 particles\n')
    sim = Simulation(50, 1, 1, 0.1, 30)
    print('Running simulation')
    show_simulation(*sim.run(1))
    
    
