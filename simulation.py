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
        infections (int): number of other particles infected
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
        self.infections = 0

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

    def infect(self, other_particle: 'Particle') -> None:
        """Infects the particle if it has not already recovered."""

        if not other_particle.is_infected and not other_particle.is_recovered:
            self.infections += 1
            other_particle.is_infected = True
            other_particle.infected_counter = self.recovery_time


class Simulation:
    """
    The Simulation object stores a collection of particles for simulation

    Attributes:
        infections (int): number of initially infected particles
        particles (List[Particle]): list of particle objects in simulation
    """

    def __init__(self, n: int, ninf: int, rad: float, speed: float, recovery_time: int) -> None:
        """
        Parameters:
            n (int): number of particles in the simulation
            ninf (inf): number of initially infected particles
            rad (float): infectious radius around particle
            speed (float): maximum speed of particles
            recovery_time (int): number of timesteps before recovery
        """

        self.infections = ninf
        self.particles = []
        self.rad = rad
        # Healthy particles
        for _ in range(n - ninf):
            x = np.random.random()
            y = np.random.random()
            vx = speed*np.random.normal(scale=0.25)
            vy = speed*np.random.normal(scale=0.25)
            self.particles.append(Particle(x, y, vx, vy, recovery_time))
        # Infected particles
        for _ in range(ninf):
            x = np.random.random()
            y = np.random.random()
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

        Parameters:
            particle (Particle): particle object to check for collisions
        """

        if particle.x < 0:
            particle.x = 0
            particle.vx *= -1
        if particle.x > 1:
            particle.x = 1
            particle.vx *= -1
        if particle.y < 0:
            particle.y = 0
            particle.vy *= -1
        if particle.y > 1:
            particle.y = 1
            particle.vy *= -1

        for other_particle in self.particles:
            if particle == other_particle:
                continue
            elif np.sqrt((particle.x - other_particle.x)**2 + (particle.y - other_particle.y)**2) <= self.rad:
                if particle.is_infected:
                    particle.infect(other_particle)

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

    def run(self, dt: float) -> Tuple[List[float], List[float], List[float], List[float], int, float]:
        """
        Runs simulation until no particles are infected

        Parameters:
            dt (float): length of the timestep

        Returns:
            x_healthy (List[float]): x-coordinates of healthy particles at each timestep
            y_healthy (List[float]): y-coordinates of healthy particles at each timestep
            x_inf (List[[float]): x-coordinates of infected particles at each timestep
            y_inf (List[float]): y-coordinates of infected particles at each timestep
            n_rec (List[int]): number of recovered particles at each timestep
        """

        x_healthy = []
        y_healthy = []
        x_inf = []
        y_inf = []
        n_rec = []
        r0 = 0

        while self.infections > 0:
            x_healthy.append([])
            y_healthy.append([])
            x_inf.append([])
            y_inf.append([])
            recovered = 0
            self.move(dt)
            for particle in self.particles:
                if particle.is_infected:
                    x_inf[-1].append(particle.x)
                    y_inf[-1].append(particle.y)  
                else:
                    if particle.is_recovered:
                        recovered += 1
                    x_healthy[-1].append(particle.x)
                    y_healthy[-1].append(particle.y)
            n_rec.append(recovered)

        infecting_particles = 0
        for particle in self.particles:
            if particle.infections > 0:
                r0 += particle.infections
                infecting_particles += 1
        r0 /= infecting_particles

        return x_healthy, y_healthy, x_inf, y_inf, n_rec, r0


def show_simulation(x_healthy: List[float], y_healthy: List[float], x_inf: List[float], y_inf: List[float], n_rec: List[int]):
    """
    Shows an animated simulation of particles until all particles are healthy. Opens 
    matplotlib window to show simulation and then plots the number of infected and 
    recovered particles over time. You may directly unroll the return values from 
    Simulation.run() with the function call show_simulation(*Simulation.run, box_size).

    Parameters
        x_healthy (List[float]): x-coordinates of healthy particles at each timestep
        y_healthy (List[float]): y-coordinates of healthy particles at each timestep
        x_inf (List[[float]): x-coordinates of infected particles at each timestep
        y_inf (List[float]): y-coordinates of infected particles at each timestep
        n_rec (List[int]): number of recovered particles at each timestep
    """

    plt.style.use('./test.mplstyle')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([])
    ax.set_yticks([])

    healthy, = ax.plot([], [], 'o')
    inf, = ax.plot([], [], 'o')

    def animate(i: int):
        healthy.set_data(x_healthy[i], y_healthy[i])
        inf.set_data(x_inf[i], y_inf[i])
        return healthy, inf

    ani = FuncAnimation(fig, animate, frames=range(len(x_healthy)), interval=100, repeat=False)
    #ani.save('test.mp4') # Uncomment if you would like save animation
    plt.show()

    timesteps = range(len(x_inf))
    n_inf = [len(i) for i in x_inf]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    inf, = ax.plot(timesteps, n_inf)
    ax.fill_between(x=timesteps, y1=n_inf, color=inf.get_color(), alpha=0.2)
    rec, = ax.plot(timesteps, n_rec)
    ax.fill_between(x=timesteps, y1=n_rec, color=rec.get_color(), alpha=0.2)
    plt.show()


if __name__ == "__main__":
    sim = Simulation(100, 5, 0.01, 1, 50)
    x_healthy, y_healthy, x_inf, y_inf, n_rec, r0 = sim.run(0.1)
    show_simulation(x_healthy, y_healthy, x_inf, y_inf, n_rec)
    print(r0)
    
    
