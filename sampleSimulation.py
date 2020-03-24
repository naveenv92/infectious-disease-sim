from simulation import *

numberOfParticles = 94*4
sim = Simulation(n=numberOfParticles, ninf=1, r=5, boxSize=2, speed=0.05, tol=0.04)
runSim = runSimulation(simulation=sim)
runSim.singleSim()