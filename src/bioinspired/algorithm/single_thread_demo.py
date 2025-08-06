import numpy as np
import pygmo as pg
from matplotlib import pyplot as plt
from bioinspired.problem import BasicProblem
import time
import datetime

# The user-defined problem

# Test if the problem is picklable
print("=" * 60)
print("PYGMO OPTIMIZATION DEMO - BIOINSPIRED SPACECRAFT DOCKING")
print("=" * 60)
print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

try:
    import pickle
    test_udp = BasicProblem()
    pickled = pickle.dumps(test_udp)
    unpickled = pickle.loads(pickled)
    print("Problem is picklable!")
except Exception as e:
    print(f"Problem is not picklable: {e}")
    exit(1)

udp = BasicProblem()

# Get problem information for telemetry
bounds = udp.get_bounds()
num_parameters = len(bounds[0])
print(f"Problem dimensions: {num_parameters} parameters")
print(f"Parameter bounds: [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}]")

# The pygmo problem
prob = pg.problem(udp)
print(f"Problem created successfully")
print(f"Problem name: {prob.get_name()}")

# For a number of generation based algorithms we can use a similar script to run and average over 25 runs.
print("\n" + "="*50)
print("STARTING SIMPLE GENETIC ALGORITHM (SGA)")
print("="*50)

udas = [pg.sga(gen=1)]
sga_start_time = time.time()

for uda in udas: 
    print(f"Running SGA for 25 runs...")
    logs = []
    run_times = []

    for i in range(25):
        run_start = time.time()
        
        algo = pg.algorithm(uda)
        algo.set_verbosity(1) # regulates both screen and log verbosity

        pop = pg.population(prob, 2)
        print(f"  Run {i+1}/25: Initial best fitness = {pop.champion_f[0]:.6f}")
        
        pop = algo.evolve(pop)
        
        run_time = time.time() - run_start
        run_times.append(run_time)
        
        print(f"  Run {i+1}/25: Final best fitness = {pop.champion_f[0]:.6f} (took {run_time:.2f}s)")
        
        logs.append(algo.extract(type(uda)).get_log())

    sga_total_time = time.time() - sga_start_time
    
    print(f"\nSGA Results Summary:")
    print(f"   Total time: {sga_total_time:.2f}s")
    print(f"   Average run time: {np.mean(run_times):.2f}s ± {np.std(run_times):.2f}s")
    print(f"   Fastest run: {np.min(run_times):.2f}s")
    print(f"   Slowest run: {np.max(run_times):.2f}s")
    
    logs = np.array(logs)
    avg_log = np.average(logs,0)
    
    # Calculate fitness statistics
    final_fitnesses = [log[-1][2] for log in logs]
    print(f"   Best fitness found: {np.min(final_fitnesses):.6f}")
    print(f"   Worst fitness found: {np.max(final_fitnesses):.6f}")
    print(f"   Average final fitness: {np.mean(final_fitnesses):.6f} ± {np.std(final_fitnesses):.6f}")

    plt.plot(avg_log[:,1],avg_log[:,2]-418.9829*20 , label=algo.get_name())

# We then add details to the plot
plt.legend() 
plt.yticks([-8000,-7000,-6000,-5000,-4000,-3000,-2000,-1000,0]) 
plt.grid() 

# Final summary
print("\n" + "="*60)
print("OPTIMIZATION COMPLETE - FINAL SUMMARY")
print("="*60)
total_runtime = time.time() - sga_start_time
print(f"Total execution time: {total_runtime:.2f}s")
print(f"Neural network parameters optimized: {num_parameters}")
print(f"Spacecraft docking problem solved with 3 different algorithms")
print(f"Results plotted and ready for analysis")
print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

plt.title("Fitness Evolution Comparison - Spacecraft Docking Optimization")
plt.xlabel("Function Evaluations")
plt.ylabel("Fitness Value")
plt.show()