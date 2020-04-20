from typing import List, Tuple
import numpy as np
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

    def move(self) -> None:
        """Moves the particle to its new position after a specified timestep."""

        self.x += self.vx
        self.y += self.vy
        
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
            vx = (-2*speed*np.random.random() + speed)/10
            vy = (-2*speed*np.random.random() + speed)/10
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
                if particle.is_infected and (particle.infected_counter <= particle.recovery_time - 7):
                    if np.random.random() < 0.7:
                        particle.infect(other_particle)

    def move(self) -> None:
        """
        Moves the simulation by one timestep.
        """

        num_infections = 0
        for particle in self.particles:
            particle.move()
            self._handle_collisions(particle)
            if particle.is_infected:
                num_infections += 1
        self.infections = num_infections

    def run(self) -> Tuple[List[float], List[float], List[float], List[float], int, float]:
        """
        Runs simulation until no particles are infected

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
            self.move()
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
        if infecting_particles > 0:
            r0 /= infecting_particles
        else:
            r0 = 0

        return x_healthy, y_healthy, x_inf, y_inf, n_rec, r0


def show_simulation(x_healthy: List[float], y_healthy: List[float], x_inf: List[float], y_inf: List[float], n_rec: List[float], n_tot: int, r0: float):
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
        n_rec (List[float]): number of recovered particles at each timestep
        n_tot (int): total number of particles
        r0 (float): reproductive ratio (R0) of infection
    """

    mpl.rcParams['font.family'] = 'Avenir, sans-serif'
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 2

    fig = plt.figure(figsize=(6,10))
    
    colors = ['#008fd5', '#fc4f30', '#e5ae38']

    # Plot of simulation box
    ax = fig.add_subplot(211)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    healthy, = ax.plot([], [], 'o', markersize=5, color=colors[0])
    inf, = ax.plot([], [], 'o', markersize=5, color=colors[1])

    # Plot of infections
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(0, len(x_healthy) - 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect(len(x_healthy))
    ax2.set_xlabel('Time', labelpad=2)
    ax2.set_ylabel('Fraction of Population', labelpad=10)

    n_recov, = ax2.plot([], [], linestyle='-', linewidth=2, color=colors[0], label='Recovered')
    n_inf, = ax2.plot([], [], linestyle='-', linewidth=2, color=colors[1], label='Infected')
    n_neverinf, = ax2.plot([], [], linestyle='-', linewidth=2, color=colors[2], label='Healthy')

    def animate(i: int):
        healthy.set_data(x_healthy[i], y_healthy[i])
        inf.set_data(x_inf[i], y_inf[i])
        ax2.collections.clear()
        n_inf.set_data(np.append(n_inf.get_data()[0], i), np.append(n_inf.get_data()[1], len(x_inf[i])/n_tot))
        ax2.fill_between(x=n_inf.get_data()[0], y1=n_inf.get_data()[1], color=n_inf.get_color(), alpha=0.2)
        n_recov.set_data(np.append(n_recov.get_data()[0], i), np.append(n_recov.get_data()[1], n_rec[i]/n_tot))
        ax2.fill_between(x=n_recov.get_data()[0], y1=n_recov.get_data()[1], color=n_recov.get_color(), alpha=0.2)
        n_neverinf.set_data(np.append(n_neverinf.get_data()[0], i), np.append(n_neverinf.get_data()[1], 1 - len(x_inf[i])/n_tot - n_rec[i]/n_tot))
        ax2.fill_between(x=n_neverinf.get_data()[0], y1=n_neverinf.get_data()[1], color=n_neverinf.get_color(), alpha=0.2)

    ax2.legend(bbox_to_anchor=(0.5, -0.18), loc=9, ncol=3, fancybox=False, frameon=False, framealpha=1.0, edgecolor='black', fontsize=16, columnspacing=0.5)
    ani = FuncAnimation(fig, animate, frames=range(len(x_healthy)), interval=100, repeat=False)
    fig.suptitle(r'R$_0$: %.2f' % r0, y=0.95)
    #ani.save('test.mp4', dpi=300) # Uncomment if you would like save animation
    plt.show()

def plot_infections(x_inf: List[float], n_rec: List[float], n_tot: int):
    """
    Parameters:
        x_inf (List[float]): x-coordinate of infected particles at each timestep
        n_rec (List[float]): number of recovered particles at each timestep
        n_tot (int): total number of particles
    """
    plt.style.use('fivethirtyeight')
    timesteps = range(len(x_inf))
    n_inf = [len(i) for i in x_inf]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    inf, = ax.plot(timesteps, [i/n_tot for i in n_inf], linewidth=2)
    ax.fill_between(x=timesteps, y1=[i/n_tot for i in n_inf], color=inf.get_color(), alpha=0.2)
    rec, = ax.plot(timesteps, [i/n_tot for i in n_rec], linewidth=2)
    ax.fill_between(x=timesteps, y1=[i/n_tot for i in n_rec], color=rec.get_color(), alpha=0.2)

    ax.set_xlim(timesteps[0], timesteps[-1])
    ax.set_ylim(0, 1.05)
    plt.show()

def multi_sim(n: int, ninf: int, rad: float, speed: float, recovery_time: int, n_runs: int) -> Tuple[List[float], List[float]]:
    """
    """
    n_inf_avg = np.zeros(50)
    n_rec_avg = np.zeros(50)
    for i in range(n_runs):
        sim = Simulation(n, ninf, rad, speed, recovery_time)
        _, _, x_inf, _, n_rec, _ = sim.run()
        n_inf = [len(j) for j in x_inf]
        
        if len(n_rec) > len(n_rec_avg):
            f1 = interp1d(np.linspace(0, 1, len(n_inf_avg)), n_inf_avg, kind='cubic')
            n_inf_avg = f1(np.linspace(0, 1, len(n_inf)))

            f2 = interp1d(np.linspace(0, 1, len(n_rec_avg)), n_rec_avg, kind='cubic')
            n_rec_avg = f2(np.linspace(0, 1, len(n_rec)))
        else:
            f1 = interp1d(np.linspace(0, 1, len(n_inf)), n_inf, kind='cubic')
            n_inf = f1(np.linspace(0, 1, len(n_inf_avg)))

            f2 = interp1d(np.linspace(0, 1, len(n_rec)), n_rec, kind='cubic')
            n_rec = f2(np.linspace(0, 1, len(n_rec_avg)))

        n_inf_avg = [sum(i) for i in zip(n_inf, n_inf_avg)]
        n_rec_avg = [sum(i) for i in zip(n_rec, n_rec_avg)]

    n_inf_avg = [i/n_runs for i in n_inf_avg]
    n_rec_avg = [i/n_runs for i in n_rec_avg]

    return n_inf_avg, n_rec_avg


if __name__ == "__main__":
    n_tot = 100
    sim = Simulation(n_tot, 10, 0.04, 0.5, 20)
    #x_healthy, y_healthy, x_inf, y_inf, n_rec, r0 = sim.run()
    #show_simulation(x_healthy, y_healthy, x_inf, y_inf, n_rec, n_tot, r0)
    n_inf_avg, n_rec_avg = multi_sim(n_tot, 5, 0.04, 0.5, 20, 10)

    plt.figure()
    plt.plot(np.linspace(0, 1, len(n_inf_avg)), n_inf_avg)
    plt.plot(np.linspace(0, 1, len(n_rec_avg)), n_rec_avg)
    plt.show()

    
    
