import random

def fitness(ch):
    return abs((ch[0] + 2*ch[1] + 3*ch[2] + 4*ch[3]) - 30)

def crossover(p1, p2, rate):
    if random.random() < rate:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]

def mutate(ch, rate):
    for i in range(len(ch)):
        if random.random() < rate:
            ch[i] = random.randint(0, 10)

def genetic_algo(size, mrate, crate, n):
    # Step 1: Generate initial population
    pop = [[random.randint(0, 10) for _ in range(n)] for _ in range(size)]
    
    print("Initial Population:")
    for ch in pop:
        print(ch)
    
    gen = 0

    while True:
        gen += 1
        # Step 2: Evaluate fitness of each chromosome
        fit = [fitness(ch) for ch in pop]

        # Find best chromosome
        best = min(pop, key=fitness)
        best_fit_val = fitness(best)


        # Step 3: Check if solution found (fitness = 0)
        if best_fit_val == 0:
            print(f"\nSolution found in {gen} generations")
            print("\nFinal Population:")
            for ch in pop:
                print(ch)
            return best

        # Step 4: Selection - pick chromosomes based on inverse of fitness (lower fitness = higher chance)
        weights = [1 / (f + 1) for f in fit]
        selected = random.choices(pop, weights=weights, k=size)

        # Step 5: Crossover - generate new population
        next_gen = []
        for _ in range(size // 2):
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2, crate)
            next_gen += [c1, c2]

        # Step 6: Mutation - apply random changes to children
        for ch in next_gen:
            mutate(ch, mrate)

        # Step 7: Replace old population with new generation
        pop = next_gen

# Run the genetic algorithm
best = genetic_algo(size=6, mrate=0.1, crate=0.8, n=4)
print("\nBest solution:", best)


import math
import random
from math import sin

def f1(x):return sin(x)
def f2(x):return -x**2
def f3(x):return -(5*x**2)+3*x+6

def best_nb(f,x,step):
	if f(x-step)>f(x):
		return x-step
	if f(x+step)>f(x):
		return x+step
	else: return x
def hillclimbing(f,x,max,step):
	for _ in range(max):
		nb= best_nb(f,x,step)
		if f(nb)>f(x):
			x=nb
		else: break
	return x

step= float(input("enter step size"))
max=int(input("enter max iterations"))
r=[(-math.pi,math.pi),(-1,1),(-100,100)]
func=[f1,f2,f3]
for i,f in enumerate(func):
	x=random.uniform(r[i][0],r[i][1])
	best_pos=round(hillclimbing(f,x,max,step),4)
	val=round(f(best_pos),4)
	print(f'For function {i + 1}, the best position is {best_pos} with a value of {val}.')

import heapq  # to use priority queue

def add_edge(graph, u, v, cost):
    graph[u].append((v, cost))
    graph[v].append((u, cost))  # undirected graph

def gbfs(graph, start, goal, heuristic):
    visited = set()
    pq = []  # priority queue: (heuristic, node, path_so_far, cost_so_far)
    heapq.heappush(pq, (heuristic[start], start, [start], 0))  # include cost_so_far

    while pq:
        print("OPEN LIST:", [(h, n) for h, n, _, _ in pq])
        print("CLOSED LIST:", visited)
        h, node, path, cost_so_far = heapq.heappop(pq)

        if node == goal:
            print("\nGoal reached!")
            print("Path:", " -> ".join(map(str, path)))
            print("Total Cost:", cost_so_far)
            return

        visited.add(node)

        for neighbor, edge_cost in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor], cost_so_far + edge_cost))

    print("Goal not reachable.")

# Main code
n = int(input("Enter number of edges: "))
max_nodes = 100
graph = [[] for _ in range(max_nodes)]

for _ in range(n):
    u, v, cost = map(int, input("Enter edge (u v cost): ").split())
    add_edge(graph, u, v, cost)

heuristic = {}
h_input = int(input("Enter number of nodes with heuristic values: "))
for _ in range(h_input):
    node, h = map(int, input("Enter node and its heuristic: ").split())
    heuristic[node] = h

start = int(input("Enter starting node for GBFS: "))
goal = int(input("Enter goal node for GBFS: "))

print("\nGreedy Best First Search Path:")
gbfs(graph, start, goal, heuristic)

import heapq  # to use priority queue

def add_edge(graph, u, v, cost):
    graph[u].append((v, cost))
    graph[v].append((u, cost))  # undirected graph

def a_star(graph, start, goal, heuristic):
    visited = set()
    pq = []  # priority queue: (f = g + h, g = cost_so_far, node, path_so_far)
    heapq.heappush(pq, (heuristic[start], 0, start, [start]))

    while pq:
        print("OPEN LIST:", [(f, n) for f, g, n, _ in pq])
        print("CLOSED LIST:", visited)
        f, g, node, path = heapq.heappop(pq)
        print("Path:", " -> ".join(map(str, path)))
        
        if node == goal:
            print("\nGoal reached!")
            print("Path:", " -> ".join(map(str, path)))
            print("Total Cost:", g)
            return

        visited.add(node)

        for neighbor, edge_cost in graph[node]:
            new_g = g + edge_cost
            new_f = new_g + heuristic[neighbor]

            # If neighbor is not visited OR has a better g (path cost)
            if neighbor not in visited:
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    print("Goal not reachable.")

# Main code
n = int(input("Enter number of edges: "))
max_nodes = 100
graph = [[] for _ in range(max_nodes)]

for _ in range(n):
    u, v, cost = map(int, input("Enter edge (u v cost): ").split())
    add_edge(graph, u, v, cost)

heuristic = {}
h_input = int(input("Enter number of nodes with heuristic values: "))
for _ in range(h_input):
    node, h = map(int, input("Enter node and its heuristic: ").split())
    heuristic[node] = h

start = int(input("Enter starting node for A* Search: "))
goal = int(input("Enter goal node for A* Search: "))

print("\nA* Search Path:")
a_star(graph, start, goal, heuristic)


import heapq  # to use priority queue

def add_edge(graph, u, v, cost):
    graph[u].append((v, cost))
    graph[v].append((u, cost))  # undirected graph

def ucs(graph, start, goal):
    visited = set()
    pq = []  # priority queue: (cost, node, path_so_far)
    heapq.heappush(pq, (0, start, [start]))  # start with path containing only the start node

    while pq:
        print("open list ", [(c, n) for c, n, _ in pq])
        print("closed list ", visited)
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        print(f"Visited Node: {node}, Cost: {cost}")
        visited.add(node)

        if node == goal:
            print("Goal reached with cost:", cost)
            print("Path:", path)
            return

        for neighbor, edge_cost in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

# Main code
n = int(input("Enter number of edges: "))
max_nodes = 100
graph = [[] for _ in range(max_nodes)]

for _ in range(n):
    u, v, cost = map(int, input("Enter edge (u v cost): ").split())
    add_edge(graph, u, v, cost)

start = int(input("Enter starting node for UCS: "))
goal = int(input("Enter goal node for UCS: "))

print("UCS Path:")
ucs(graph, start, goal)

def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)  # undirected graph

def dls(graph, node, target, limit, visited):
    if node == target:
        print(node, end=" ")
        return True

    if limit <= 0:
        return False

    visited[node] = True
    print(node, end=" ")

    for neighbor in graph[node]:
        if not visited[neighbor]:
            if dls(graph, neighbor, target, limit - 1, visited):
                return True

    return False

def dfid(graph, start, target, max_depth):
    for depth in range(max_depth + 1):
        visited = [False] * len(graph)
        print(f"\nTrying depth limit = {depth}: ", end="")
        if dls(graph, start, target, depth, visited):
            print(f"\n Found target {target} at depth {depth}")
            return
    print(f"\n Target {target} not found within depth {max_depth}")

# Main code
n = int(input("Enter number of edges: "))
max_nodes = 100
graph = [[] for _ in range(max_nodes)]

for _ in range(n):
    u, v = map(int, input("Enter edge (u v): ").split())
    add_edge(graph, u, v)

start = int(input("Enter starting node: "))
target = int(input("Enter target node to find: "))
max_depth = int(input("Enter maximum depth limit: "))

dfid(graph, start, target, max_depth)

def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)  

def dls(graph, visited, node, limit):
    if limit < 0:
        return
    visited[node] = True
    print(node, end=" ")

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dls(graph, visited, neighbor, limit - 1)

# Main code
n = int(input("Enter number of edges: "))
max = 100
graph = [[] for _ in range(max)]
visited = [False] * max

for _ in range(n):
    edge = input("Enter edge (u v): ").split()
    u=int(edge[0])
    v=int(edge[1])
    add_edge(graph, u, v)

start = int(input("Enter starting node for DLS: "))
limit = int(input("Enter depth limit: "))

print("DLS Path:", end=" ")
dls(graph, visited, start, limit)

from collections import deque  # to use queue

def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)  # undirected graph

def bfs(graph, visited, start):
    queue = deque()        # create a queue
    visited[start] = True  # mark start node as visited
    queue.append(start)    # push start node into queue

    while queue:
        node = queue.popleft()  # remove front element
        print(node, end=" ")

        for neighbor in graph[node]:  # check neighbors
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

# Main code
n = int(input("Enter number of edges: "))
max = 100  
graph = [[] for _ in range(max)]
visited = [False] * max

for _ in range(n):
    edge = input("Enter edge (u v): ").split()
    u = int(edge[0])
    v = int(edge[1])
    add_edge(graph, u, v)

start = int(input("Enter starting node for BFS: "))
print("BFS Path:", end=" ")
bfs(graph, visited, start)
