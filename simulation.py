# Graphical and numerical required libraries
import matplotlib.pyplot as plt
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
	def __init__(self, x=0, y=0, vx=1, vy=1, r=2):
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
		self._x += self._vx*dt
		self._y += self._vy*dt

		if self._isInfected == True:
			self._infectedCounter -= 1
			if self._infectedCounter == 0:
				self._isInfected = False
				self._isRecovered = True

	## Infects particle if it not yet been infected and recovered, and sets a number of timesteps until recovery
	def infect(self):
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
	def infectOtherPerson(self):
		self._numberOfInfections += 1

	## Returns the number of other particles infected
	def numOfInfections(self):
		return self._numberOfInfections


class Simulation:

	def __init__(self, n=10, ninf=1, r=10, boxSize=1, speed=1, tol=0.1):
		self._n = n
		self._particles = []
		self._tol = tol
		self._boxSize = boxSize
		self._speed = speed
		for _ in range(n):
			vx = self._speed*np.random.random()
			vy = self._speed*np.random.random()
			direction_x = 2*(np.random.randint(2) - 0.5)
			direction_y = 2*(np.random.randint(2) - 0.5)
			self._particles.append(Particle(self._boxSize*np.random.random(), self._boxSize*np.random.random(), direction_x*vx, direction_y*vy, r))

		for i in range(ninf):
			self._particles[i].infect()

	def n(self):
		return self._n

	def getBoxSize(self):
		return self._boxSize

	def x_coords(self):
		x_coords = []
		for i in self._particles:
			if not i.isInfected():
				x_coords.append(i.x())
		return x_coords

	def x_coords_inf(self):
		x_coords_inf = []
		for i in self._particles:
			if i.isInfected():
				x_coords_inf.append(i.x())
		return x_coords_inf

	def y_coords(self):
		y_coords = []
		for i in self._particles:
			if not i.isInfected():
				y_coords.append(i.y())
		return y_coords

	def y_coords_inf(self):
		y_coords_inf = []
		for i in self._particles:
			if i.isInfected():
				y_coords_inf.append(i.y())
		return y_coords_inf

	def radii(self):
		radii = []
		for i in self._particles:
			radii.append(i.r())
		return radii

	def move(self, dt=1):
		for i in self._particles:
			i.move(dt)
		self.manageCollisions()

	def manageCollisions(self):
		for i in self._particles:
			if i.x() <= 0 or i.x() >= self._boxSize:
				i.set_vx(-i.vx())
			if i.y() <=0 or i.y() >= self._boxSize:
				i.set_vy(-i.vy())

		for i in range(len(self._particles)):
			for j in range(i + 1, len(self._particles)):
				particle_1 = self._particles[i]
				particle_2 = self._particles[j]
				if particle_1.x() - self._tol <= particle_2.x() <= particle_1.x() + self._tol and particle_1.y() - self._tol <= particle_2.y() <= particle_1.y() + self._tol:
					if particle_1.isInfected() and not particle_2.isInfected() and not particle_2.isRecovered():
						particle_1.infectOtherPerson()
						particle_2.infect()
					if particle_2.isInfected() and not particle_1.isInfected() and not particle_1.isRecovered():
						particle_2.infectOtherPerson()
						particle_1.infect()


	def statistics(self):
		countInf = 0
		countRec = 0
		countNI = 0
		avgInfections = 0
		numOfInfectors = 1
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
				plt.scatter(self._simulation.x_coords(), self._simulation.y_coords(), s=self._simulation.radii()[0]**2, color='red')
				plt.scatter(self._simulation.x_coords_inf(), self._simulation.y_coords_inf(), s=self._simulation.radii()[0]**2, color='black')
				plt.xticks([])
				plt.yticks([])
				plt.xlim(-0.01, self._simulation.getBoxSize() + 0.01)
				plt.ylim(-0.01, self._simulation.getBoxSize() + 0.01)
				plt.draw()
				plt.pause(0.01)
				plt.clf()

			self._simulation.move()

		plt.ioff()
		plt.close()

		self._numNI = numNI
		self._numInf = numInf
		self._numRec = numRec

		print(self._simulation.statistics()[3])

		if plot:
			fig = plt.figure(figsize=(5,5))
			ax = fig.add_axes([0, 0, 1, 1])

			ax.plot(np.arange(0, timesteps, 1), numInf, linewidth=2, label='Infected')
			ax.plot(np.arange(0, timesteps, 1), numRec, linewidth=2, label='Recovered')

			ax.set_xlabel('Timesteps')
			ax.set_ylabel('Number')

			ax.set_ylim(0, self._simulation.n())

			ax.legend()

			plt.savefig('simulation.png', dpi=300, bbox_inches='tight')
			plt.show()

	def multiSim(self, numOfSims=5, plot=True):

		totalNumNI = np.array([])
		totalNumInf = np.array([])
		totalNumRec = np.array([])

		for i in range(numOfSims):
			self.singleSim(showSim=False, plot=False)
			totalNumNI += self._numNI
			totalNumInf += self._numInf
			totalNumRec += self._numRec

		totalNumNI = [i/numOfSims for i in totalNumNI]
		totalNumInf = [i/numOfSims for i in totalNumInf]
		totalNumRec = [i/numOfSims for i in totalNumRec]

		if plot:
			fig = plt.figure(figsize=(5,5))
			ax = fig.add_axes([0, 0, 1, 1])

			ax.plot(np.arange(0, timesteps, 1), totalNumInf, linewidth=2, label='Infected')
			ax.plot(np.arange(0, timesteps, 1), totalNumRec, linewidth=2, label='Recovered')

			ax.set_xlabel('Timesteps')
			ax.set_ylabel('Number')

			ax.set_ylim(0, self._simulation.n())

			ax.legend()

			plt.savefig('simulation.png', dpi=300, bbox_inches='tight')
			plt.show()