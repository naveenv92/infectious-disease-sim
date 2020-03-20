from simulation import *

numberOfPeople = 400
sim = Simulation(n=numberOfPeople, ninf=10, r=5, boxSize=2, speed=0.5, tol=0.05)
runSim = runSimulation(simulation=sim)
runSim.singleSim()