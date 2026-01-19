import numpy as np
import networkx as nx
import copy

np.random.seed(0)

#modified by input
n = 20
m = 50
req = np.full((n ,n), 0) #nodes are 0 indexed
edge_list = np.full((3, m), 0) #edges are 0 indexed
costs = np.full(m, 0)

#not modified by input
tol_x = 1e-2
tol_lamda = 1e-3
tol_det=1e-7
max_iter=100


def separation_oracle(x, lamda, edges_found): #to check if a x is feasible for the fractional LP instance

    #checking if each xi is in the range [0,1]
    for i in range(m):
        if(x[i] < -tol_x) : #xi >= 0
            arr = np.zeros(m)
            arr[i] = -1
            return (False, arr)
        if(x[i] > 1 + tol_x) : #xi <= 1
            arr = np.zeros(m)
            arr[i] = 1
            return (False, arr)

    #checking if sum of cost(i) * x(i) <= lamda   
    const = 0
    for i in range(m) :
        const += (costs[i] * x[i])
    if(const > lamda):
        arr = copy.deepcopy(costs)
        return (False, arr)

    #checking if for edges already found, xi is 1
    for e in edges_found : 
        if(x[e] < 1 - tol_x) : #xe >= 1
            arr = np.zeros(m)
            arr[e] = -1
            return (False, arr)

    #creating a directed graph from undirected graph with edge capacities as xi             
    G = nx.DiGraph()
    edge_cap = dict()
    for i in range(n) :
        G.add_node(i)

    for i in range(m) :
        if(x[i] != 0):
            G.add_edge(edge_list[0][i], edge_list[1][i], capacity = x[i])
            edge_cap[(edge_list[0][i], edge_list[1][i])] = x[i]
            G.add_edge(edge_list[1][i], edge_list[0][i], capacity = x[i])
            edge_cap[(edge_list[1][i], edge_list[0][i])] = x[i]

    for source in range(0, n, 1) :
        for sink in range(0, n, 1) :
            if(sink == source) :
                continue
            #computing the source-sink flow
            (flow_value, flow_dict) = nx.maximum_flow(G, source, sink, flow_func = nx.algorithms.flow.edmonds_karp)

            #creating residual graph of this flow
            flow_cap = dict()
            for key1, value1 in flow_dict.items():
                for key2, value2 in value1.items():
                    flow_cap[(key1, key2)] = value2

            Gres = nx.DiGraph()
            for i in range(n):
                Gres.add_node(i)

            for e, cap in edge_cap.items():
                if((e not in flow_cap) or (flow_cap[e] < cap - tol_x)):
                    Gres.add_edge(e[0], e[1], capacity = 1)
            
            #finding the reachable and non-reachable sets for this flow
            reachable = nx.descendants(Gres, source)
            reachable.add(source)
            non_reachable = set(Gres.nodes()) - reachable

            coeff = np.zeros(m)
            if(flow_value < req[source][sink] - 1e-2) : #checking if some (source,sink) flow is violating requirement
                for i in range(m) :
                    if(((edge_list[0][i] in reachable) and (edge_list[1][i] in non_reachable)) 
                       or ((edge_list[1][i] in reachable) and (edge_list[0][i] in non_reachable))) :
                        if(not (i in edges_found)):
                            coeff[i] = -1 
                return (False, coeff) #return the violated cut
    
    return (True, None)

def feasible_solution(edges_found) : #to check if edges found till now are feasible

    #checking if we only consider the edges in edges_found, is the obtained solution feasible or not

    G = nx.DiGraph()
    for i in range(n) :
        G.add_node(i)
    for i in range(m) :
        if(i in edges_found):
            G.add_edge(edge_list[0][i], edge_list[1][i], capacity = 1)
            G.add_edge(edge_list[1][i], edge_list[0][i], capacity = 1)
    
    for source in range(0, n, 1) :
        for sink in range(0, n, 1) :
            if(sink == source) :
                continue
            (flow_value, flow_dict) = nx.maximum_flow(G, source, sink, flow_func = nx.algorithms.flow.edmonds_karp)
            if(flow_value < req[source][sink] - 1e-2) :            
                return False
            
    return True

def optimal_LP_soln(): #to find the optimal value of the fractional LP given the graph instance
    #finding the optimal value of LP when we have all edges
    low = 0
    high = 0
    for i in range(m):
        high += costs[i]
        
    while(abs(high - low) > tol_lamda):
        lamda = (low + high) / 2
        found = False
        x = [np.random.random() for i in range(m)]
        P = np.eye(m) * 1000
        for iteration in range(max_iter):
            feasible, hyperplane = separation_oracle(x, lamda, dict())
            if feasible:
                found = True
                break
            
            a = hyperplane
            norm = np.sqrt(a.T @ P @ a)
            a = a / norm
                        
            x = x - (1 / (m + 1)) * (P @ a)
            P = (m**2 / (m**2 - 1)) * (P - (2 / (m + 1)) * (P @ np.outer(a, a) @ P))
            
            if np.linalg.det(P) < tol_det:
                print(f"Ellipsoid volume too small, no solution found for lambda : {lamda}")
                break
        
        if(found) :
            high = lamda
        else :
            low = lamda
        
    return ((low + high) / 2)
    
def ellipsoid_method(): #to get the optimal value of objective function

    edges_found = set()

    while(not (feasible_solution(edges_found))) :
        low = 0
        high = 0
        for i in range(m):
            high += costs[i]
            
        solution_found = False

        while(abs(high - low) > tol_lamda):
            lamda = (low + high) / 2
            print(f"Starting for lamda = {lamda}")
            found = False
            x = [np.random.random() for i in range(m)]
            for e in edges_found:
                x[e] = 1.0 #just fix all those edges which have come to 1
            P = np.eye(m) * 1000

            for iteration in range(max_iter):
                # Check feasibility using separation oracle
                feasible, hyperplane = separation_oracle(x, lamda, edges_found)
                if feasible:
                    print(f"Solution found for {lamda}")
                    solution_found = True
                    found = True
                    break
                
                a = hyperplane
                norm = np.sqrt(a.T @ P @ a)
                if(norm < tol_x):
                    print("Norm value is too small")

                a = a / norm
                                
                x = x - (1 / (m + 1)) * (P @ a)
                P = (m**2 / (m**2 - 1)) * (P - (2 / (m + 1)) * (P @ np.outer(a, a) @ P))
                
                if np.linalg.det(P) < tol_det:
                    print(f"Ellipsoid volume too small, no solution found for lambda : {lamda}")
                    break
            
            if(found) :
                high = lamda   
            else :
                low = lamda
        
        cnt = 0
        for i in range(m):
            if(x[i] > (0.50 - tol_x)):
                cnt += 1
                edges_found.add(i)
                costs[i] = 0
                x[i] = 1.0

        if(cnt == 0):
            print(f"Cnt value is 0 and max of x is {np.max(x)}")
            max_id = np.argmax(x)
            edges_found.add(max_id)
            costs[max_id] = 0
            x[max_id] = 1.0            
        
        if(not solution_found) :
            print("No solution exists for the problem")
            break

    return edges_found

# Example usage
if __name__ == "__main__":
    fp = open("random_input.txt",'r')
    line = fp.readline().split(' ')
    n = int(line[0])
    m = int(line[1])
    req = np.full((n ,n), 0) #nodes are 0 indexed
    edge_list = np.full((3, m), 0) #edges are 0 indexed
    costs = np.full(m, 0)

    #print(n, m)
    for e in range(m) :
        line = fp.readline().split(' ')
        edge_list[0][e] = int(line[0])
        edge_list[1][e] = int(line[1])
        edge_list[2][e] = int(line[2])
        costs[e] = edge_list[2][e]

    #print(edge_list)

    for i in range(0, n, 1) :
        line = fp.readline().split(' ')
        for j in range(0, n, 1) :
            req[i][j] = int(line[j])

    #print(req)

    edges_found = ellipsoid_method()
    print("Optimal value of lambda is ", optimal_LP_soln())
    print("Is solution feasible? ", feasible_solution(edges_found))
    print("Number of edges in optimal soln : ", len(edges_found))
    solution_cost = 0
    for i in edges_found:
        solution_cost += edge_list[2][i]
    print("Cost of solution is : ", solution_cost)

