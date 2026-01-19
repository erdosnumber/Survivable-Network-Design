import random
import sys
import numpy as np

random.seed(0)

def generate_random_instance(n, max_cost=10, max_requirement=2, edge_probability=0.8):

    vertices = n
    edges = []
    connectivity_requirements = [[0 for i in range(n)] for j in range(n)]

    #we assume that (i,j) is an edge in the graph with probability edge_probability and has cost in the 
    #range [1, max_cost]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_probability:
                cost = random.randint(1, max_cost)
                edges.append([i, j, cost])

    #We create a n x n symmetric requirement matrix where each entry is in [0,max_requirement]
    for i in range(n):
        connectivity_requirements[i][i] = 0
        for j in range(i + 1, n):
            if(random.random() < 0.3) : #with 0.3 probability we let an entry to be 0
                connectivity_requirements[i][j] = int(0)
                connectivity_requirements[j][i] = int(0)
            else: #with reamaining probability we sample a random entry in [1, max_requirement]
                requirement = random.randint(1, max_requirement)
                connectivity_requirements[i][j] = requirement
                connectivity_requirements[j][i] = requirement

    return {
        "vertices": vertices,
        "edges": edges,
        "connectivity_requirements": connectivity_requirements,
    }

n = int(sys.argv[1])
instance = generate_random_instance(n)

fp = open("random_input.txt", 'w')

fp.write(str(instance["vertices"])+" "+str(len(instance["edges"]))+"\n")
for e in instance["edges"] :
    fp.write(str(e[0])+" "+str(e[1])+" "+str(e[2])+"\n")

for i in range(n):
    for j in range(n):
        fp.write(str(instance["connectivity_requirements"][i][j])+" ")
    fp.write("\n")


