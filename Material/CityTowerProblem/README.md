# Cell Tower Location Problem:
The solution to find out the optimal location of cell tower sites is discussed here by keeping constrain over budget and population. To simulate the given problem, first random population is generated. The whole population is divided into regions using K-Means clustering. The centroid of each region is used as a potential site to plant to the cell tower. Out of all potential sites, the goal is to obtain the optimal locations such that total cost to plant the towers does not exceed the budget, and the maximum population is also served. The formulation is obtained by an instance of mixed-integer linear programming and solved by branch and bound technique using Gurobi's python based solver.

# Simulate:
*  To simulate run command below,

python TowerPlanning.py 'Number of Regions'


-> Result will be stores in ./result/'Number of Regions' Folder <br/>
->  Summary of the result will be given ./Result.csv <br/>

*  To generate plot time complexity run,

python PlotResult.py

*  To Run the simulation for many input run,
chmod +x Simulation.sh
./Simulation.sh

* To Run the jupyter notebook file use TowerPlanningJupyter.ipynb file

Note: Change the appropriate input accordingly as per need 

# File Directory:
-ME17S301 <br/>

-- CODE  <br/>
--- TowerPlanning.py  <br/>
--- PlotResult.py  <br/>
--- Simulation.sh  <br/>
--- result - all the result files\folders will be generated here  <br/>
--- Result.csv -Summary of simulation run with execution time  <br/>

-- ME17S301_Report.pdf  <br/>

-- README.md  <br/>


# Dependencie

* numpy
* matplotlib
* pandas
* Gurobi Optimizer
* scikit learn

