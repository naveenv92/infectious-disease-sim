## Graphical and numerical required libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

## Particle class -- keeps track of position, velocity, infection state, and number of other people infected
class Particle:

	## Initialize particle with location and velocity, and set default infected state to False
	# _x -- x-position
	# _y -- y-position
	# _vx -- velocity in the x-direction
	# _vy -- velocity in the y-direction
	# _r -- size of point shown in simulation (only if showing simulation)
	# _isInfected -- boolean representing if particle in infected
	# _isRecovered -- boolean representing if particle has recovered (and therefore, cannot be infected again)
	# _numberOfInfections -- number of other particles infected
	def __init__(self, x=0, y=0, vx=1, vy=1, r=5):
		self._x = x
		self._y = y
		self._vx = vx
		self._vy = vy
		self._r = r
		self._isInfected = False
		self._isRecovered = False
		self._numberOfInfections = 0

	## Returns the x-position of the particle
	def x(self):
		return self._x

	## Sets the x-position of the particle
	def set_x(self, x):
		self._x = x

	## Returns the y-position of the particle
	def y(self):
		return self._y

	## Sets the y-position of the particle
	def set_y(self, y):
		self._y = y

	## Returns the velocity component in the x-direction of the particle
	def vx(self):
		return self._vx

	## Sets the velocity component in the x-direction of the particle
	def set_vx(self, vx):
		self._vx = vx

	## Returns the velocity component in the y-direction of the particle
	def vy(self):
		return self._vy

	## Sets the velocity component in the y-direction of the particle
	def set_vy(self, vy):
		self._vy = vy

	## Returns the size of the point shown in the simulation
	def r(self):
		return self._r

	## Sets the size of the point shown in the simulation
	def set_r(self, r):
		self._r = r

	## Moves the particle by a timestep (default is 1) along its velocity vector and updates its position
	# If infected, counts down timesteps until recovery
	def move(self, dt=1):
		# Update x and y coordinates
		self._x += self._vx*dt
		self._y += self._vy*dt

		# If infected, increment countdown until recovery
		if self._isInfected == True:
			self._infectedCounter -= 1
			if self._infectedCounter == 0:
				self._isInfected = False
				self._isRecovered = True

	## Infects particle if it not yet been infected and recovered, and sets a number of timesteps until recovery
	def infect(self):
		# Infect if particle has never been infected (not recovered)
		if self._isRecovered == False and self._isInfected == False:
			self._isInfected = True
			self._infectedCounter = 30

	## Returns the infection state of the particle (True or False)
	def isInfected(self):
		return self._isInfected

	## Returns the recovered state of the particle (True or False)
	def isRecovered(self):
		return self._isRecovered

	## Increment the number of other particles infected
	def infectOtherParticle(self):
		self._numberOfInfections += 1

	## Returns the number of other particles infected -- used to calculate R_0 value
	def numOfInfections(self):
		return self._numberOfInfections


## Simulation class -- creates a simulation of a collection of Particle objects and keeps track of important simulation statistics
class Simulation:

	## Initialize simulation with a number of particles
	# _n -- number of particles in the simulation
	# _particles -- list of Particle objects in the simulation
	# _tol -- tolerance around each particle for a collision
	# _boxSize -- size of simulation box in each direction (i.e. 1 x 1, 2 x 2, 3 x 3, etc.)
	# _speed -- maximum value of each velocity component (total speed is bounded between 0 and sqrt(2)*_speed)
	def __init__(self, n=100, ninf=1, r=5, boxSize=1, speed=1, tol=0.1):
		self._n = n
		self._particles = []
		self._tol = tol
		self._boxSize = boxSize
		self._speed = speed
		
		# Populate _particles with _n particles with random position and velocity
		for _ in range(n):
			vx = self._speed*np.random.random()
			vy = self._speed*np.random.random()
			
			# Randomize the direction of each velocity component by choosing either -1 or 1
			direction_x = 2*(np.random.randint(2) - 0.5)
			direction_y = 2*(np.random.randint(2) - 0.5)
			self._particles.append(Particle(self._boxSize*np.random.random(), self._boxSize*np.random.random(), direction_x*vx, direction_y*vy, r))

		# Infect a number of particles based on input value ninf
		for i in range(ninf):
			self._particles[i].infect()

	## Returns the number of particles
	def n(self):
		return self._n

	## Returns the size of the simulation box
	def getBoxSize(self):
		return self._boxSize

	## Returns arrays of x and y coordinates of non-infected particles
	def coords(self):
		x_coords = []
		y_coords = []
		for i in self._particles:
			if not i.isInfected():
				x_coords.append(i.x())
				y_coords.append(i.y())
		return x_coords, y_coords

	## Returns arrays of x and y coordinates of infected particles
	def coords_inf(self):
		x_coords_inf = []
		y_coords_inf = []
		for i in self._particles:
			if i.isInfected():
				x_coords_inf.append(i.x())
				y_coords_inf.append(i.y())
		return x_coords_inf, y_coords_inf

	## Returns size of particle for simulation (all are same size so return the first value)
	def radii(self):
		return self._particles[0].r()

	## Move each particle and manage any collisions
	def move(self, dt=1):

		## Manage collisions between walls and other particles
		def manageCollisions():
			
			# If particles hit the walls of the simulation box, bounce off
			for i in self._particles:
				if i.x() <= 0 or i.x() >= self._boxSize:
					i.set_vx(-i.vx())
				if i.y() <=0 or i.y() >= self._boxSize:
					i.set_vy(-i.vy())

			# Iterate through particles and manage collisions between pairs of particles
			for i in range(len(self._particles)):
				for j in range(i + 1, len(self._particles)):
					particle_1 = self._particles[i]
					particle_2 = self._particles[j]
					if particle_1.x() - self._tol <= particle_2.x() <= particle_1.x() + self._tol and particle_1.y() - self._tol <= particle_2.y() <= particle_1.y() + self._tol:
						# Infect other particle if current particle is infected and the other particle is not infected or recovered
						if particle_1.isInfected() and not particle_2.isInfected() and not particle_2.isRecovered():
							particle_1.infectOtherParticle()
							particle_2.infect()
						if particle_2.isInfected() and not particle_1.isInfected() and not particle_1.isRecovered():
							particle_2.infectOtherParticle()
							particle_1.infect()

		for i in self._particles:
			i.move(dt)
		manageCollisions()

	## Returns important simulation statistics -- number not infected, number infected, number recovered, and average number infected (~R_0 value)
	def statistics(self):
		countInf = 0
		countRec = 0
		countNI = 0
		avgInfections = 0
		numOfInfectors = 1
		
		# Check status of each particle - for R_0 only count particles that have infected other particles
		for i in self._particles:
			if i.isInfected():
				countInf += 1
			elif i.isRecovered():
				countRec += 1
			else:
				countNI += 1
			if i.numOfInfections() > 0:
				avgInfections += i.numOfInfections()
				numOfInfectors += 1

		avgInfections = avgInfections/numOfInfectors

		return [countNI, countInf, countRec, avgInfections]


class runSimulation:

	def __init__(self, simulation):
		self._simulation = simulation
		self._numNI = np.array([])
		self._numInf = np.array([])
		self._numRec = np.array([])

	def singleSim(self, showSim=True, plot=True):
		timesteps = 0
		numNI = np.array([])
		numInf = np.array([])
		numRec = np.array([])

		mpl.rcParams['font.family'] = 'Avenir'
		plt.rcParams['font.size'] = 18
		plt.rcParams['axes.linewidth'] = 3
		plt.ion()

		currentInf = self._simulation.statistics()[1]

		while currentInf > 0:
			currentStats = self._simulation.statistics()
			numNI = np.append(numNI, currentStats[0])
			numInf = np.append(numInf, currentStats[1])
			numRec = np.append(numRec, currentStats[2])
			currentInf = currentStats[1]
			timesteps += 1
			
			if showSim:
				coords = self._simulation.coords()
				coords_inf = self._simulation.coords_inf()
				r = self._simulation.radii()**2
				plt.scatter(coords[0], coords[1], s=r, color='#1e90ff')
				plt.scatter(coords_inf[0], coords_inf[1], s=r, color='#ff4500')
				plt.xticks([])
				plt.yticks([])
				plt.xlim(-0.01, self._simulation.getBoxSize() + 0.01)
				plt.ylim(-0.01, self._simulation.getBoxSize() + 0.01)
				plt.title('Infected: ' + str(int(numInf[-1])) + '  Recovered: ' + str(int(numRec[-1])))
				plt.draw()
				plt.pause(0.01)
				plt.clf()

			self._simulation.move()

		plt.ioff()
		plt.close()

		self._numNI = numNI
		self._numInf = numInf
		self._numRec = numRec

		r0_val = 'R_0 value: ' + str(self._simulation.statistics()[3])

		np.savetxt('Simulation_Statistics.csv', np.c_[self._numNI, self._numInf, self._numRec], delimiter=',', header='Not Infected, Infected, Recovered', footer=r0_val, comments='')

		if plot:
			plt.rcParams['axes.linewidth'] = 2

			fig = plt.figure(figsize=(6,4))
			ax = fig.add_axes([0, 0, 1, 1])

			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)

			ax.xaxis.set_tick_params(which='major', size=10, width=2)
			ax.yaxis.set_tick_params(which='major', size=10, width=2)

			ax.fill_between(np.arange(0, timesteps, 1), numRec, 0, linewidth=0, color='#85c3ff', alpha=0.2)
			ax.plot(np.arange(0, timesteps, 1), numRec, linewidth=3, color='#1e90ff', label='Recovered')
			ax.fill_between(np.arange(0, timesteps, 1), numInf, 0, linewidth=0, color='#ffa381', alpha=0.2)
			ax.plot(np.arange(0, timesteps, 1), numInf, linewidth=3, color='#ff4500', label='Infected')

			ax.set_xlabel('# of timesteps', labelpad=10)
			ax.set_ylabel('# of particles', labelpad=10)

			ax.set_xlim(-0.01*timesteps, 1.01*timesteps)
			ax.set_ylim(-0.01*self._simulation.n(), 1.01*self._simulation.n())

			ax.legend(loc=2, frameon=False)

			plt.savefig('Simulation.png', dpi=300, bbox_inches='tight')
			plt.show()