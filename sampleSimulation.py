from simulation import *

numberOfPeople = 376
sim = Simulation(n=numberOfPeople, ninf=5, r=5, boxSize=2, speed=0.05, tol=0.02)
runSim = runSimulation(simulation=sim)
runSim.singleSim()