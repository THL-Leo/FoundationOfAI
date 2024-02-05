##############################################
# CSCI 561 SP 24 HW 1
# Name: Leo Lee
# USC ID: 8190296984
##############################################
# import time
import heapq

class Node:
    def __init__(self, location, parent, momentum, cost):
        self.location = location
        self.parent = parent
        self.momentum = momentum
        self.cost = cost

    def __lt__(self, other):
        # This is used for comparison in heapq
        return self.cost < other.cost

# def is_valid(energy_required, momentum, energy_limit):
#     if momentum + energy_limit >= energy_required or energy_required <= 0:
#         return True
#     return False

# def required_energy(locations, current, destination):
    # return locations[destination][2] - locations[current][2]

def expand(locations, paths, node, algo):
    s = node.location
    momentum = node.momentum
    for path in paths[s]:
        cost = 0
        s_prime = path
        energy_required = locations[s_prime][2] - locations[s][2]
        if momentum + energy_limit >= energy_required or energy_required <= 0:
            if algo == 'BFS':
                cost = node.cost + 1
            elif algo == 'UCS':
                x1, y1 = locations[s][:2]
                x2, y2 = locations[s_prime][:2]
                cost = node.cost + ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            elif algo == 'A*':
                x1, y1, z1 = locations[s]
                x2, y2, z2 = locations[s_prime]
                cost = node.cost + ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        # energy = required_energy(locations, s, s_prime)
            yield Node(s_prime, node, max(0, energy_required * -1), cost)

def replace_node(L, node):
    temp = []
    for n in L:
        if n[2].location == node.location and n[2].parent == node.parent and node.cost < n[2].cost:
            heapq.heappush(temp, (node.cost, node))
        else:
            heapq.heappush(temp, n)
    return temp

def BFS(locations, paths):
    start_node = Node('start', None, 0, 0)
    visited = {'start': start_node}
    frontier = []
    frontier.append(start_node)
    while frontier:
        current_node = frontier.pop(0)
        if current_node.location == 'goal':
            return current_node
        for neighbor_node in expand(locations, paths, current_node, 'BFS'):
            s = neighbor_node.location
            if s not in visited or neighbor_node.momentum > visited[s].momentum:
                visited[s] = neighbor_node
                frontier.append(neighbor_node)
    return ['FAIL']

def UCS(locations, paths):
    start_node = Node('start', None, 0, 0)
    visited = {}  # CLOSED QUEUE
    frontier = []
    frontier_map = {}
    heapq.heappush(frontier, (0, start_node))
    frontier_map['start'] = (0, start_node)

    while frontier:
        elem = heapq.heappop(frontier)
        current_node = elem[1]

        if current_node.location == 'goal':
            return current_node

        if current_node.location == 'start':
            visited['start'] = elem
        else:
            visited[current_node.parent.location + ' ' + current_node.location] = elem

        for neighbor_node in expand(locations, paths, current_node, 'UCS'):
            s = neighbor_node.parent.location + ' ' + neighbor_node.location
            inVisited = s in visited
            inFrontier = s in frontier_map
            if not inVisited and not inFrontier:
                heapq.heappush(frontier, (neighbor_node.cost, neighbor_node))
                frontier_map[s] = (neighbor_node.cost, neighbor_node)
            elif inFrontier: 
                if neighbor_node.cost < frontier_map[s][1].cost:
                    frontier_map[s] = (neighbor_node.cost, neighbor_node)
                    frontier = replace_node(frontier, neighbor_node)
            elif inVisited:
                if neighbor_node.cost < visited[s][0]:
                    del visited[s]
                    frontier_map[s] = (neighbor_node.cost, neighbor_node)
                    heapq.heappush(frontier, (neighbor_node.cost, neighbor_node))
    return ['FAIL']

def A_star(locations, paths):
    start_node = Node('start', None, 0, 0)
    visited = {}  # CLOSED QUEUE
    frontier = []
    frontier_map = {}
    heapq.heappush(frontier, (0, start_node))
    frontier_map['start'] = (0, start_node)

    while frontier:
        elem = heapq.heappop(frontier)
        current_node = elem[1]

        if current_node.location == 'goal':
            return current_node

        if current_node.location == 'start':
            visited['start'] = elem
        else:
            visited[current_node.parent.location + ' ' + current_node.location] = elem

        for neighbor_node in expand(locations, paths, current_node, 'A*'):
            s = neighbor_node.parent.location + ' ' + neighbor_node.location
            inVisited = s in visited
            inFrontier = s in frontier_map
            if not inVisited and not inFrontier:
                heapq.heappush(frontier, (neighbor_node.cost, neighbor_node))
                frontier_map[s] = (neighbor_node.cost, neighbor_node)
            elif inFrontier:
                if neighbor_node.cost < frontier_map[s][1].cost:
                    frontier_map[s] = (neighbor_node.cost, neighbor_node)
                    frontier = replace_node(frontier, neighbor_node)
            elif inVisited:
                if neighbor_node.cost < visited[s][0]:
                    del visited[s]
                    frontier_map[s] = (neighbor_node.cost, neighbor_node)
                    heapq.heappush(frontier, (neighbor_node.cost, neighbor_node))
    return ['FAIL']

if __name__ == '__main__':
    #exec_time = time.time()
    method, energy_limit, N, locations, M, paths = '', -1, 0, {}, 0, {}
    with open('input.txt', 'r') as input:
        method = input.readline().rstrip('\n')
        energy_limit = int(input.readline().rstrip('\n'))
        N = int(input.readline().rstrip('\n'))
        for _ in range(N):
            name, x, y, z = input.readline().rstrip('\n').split(' ')
            x = int(x)
            y = int(y)
            z = int(z)
            locations[name] = (x, y, z)
        M = int(input.readline().rstrip('\n'))
        for _ in range(M):
            loc_1, loc_2 = input.readline().rstrip('\n').split(' ')
            if paths.get(loc_1) is None:
                paths[loc_1] = [loc_2]
            else:
                paths[loc_1].append(loc_2)
            if paths.get(loc_2) is None:
                paths[loc_2] = [loc_1]
            else:
                paths[loc_2].append(loc_1)

    if method == 'BFS':
        # cProfile.run('BFS(locations=locations, paths=paths)')
        output = BFS(locations=locations, paths=paths)
        f = open('output.txt', 'w')
        if isinstance(output, list):
            f.write(' '.join(output))
            f.write('\n')
            f.close()
        else:
            L = []
            while output.parent is not None:
                L.append(output.location)
                output = output.parent

            L.append(output.location)
            L.reverse()
            f.write(' '.join(L))
            f.write('\n')
            f.close()

    if method == 'UCS':
        # cProfile.run('UCS(locations=locations, paths=paths)')
        output = UCS(locations=locations, paths=paths)
        f = open('output.txt', 'w')
        if isinstance(output, list):
            f.write(' '.join(output))
            f.write('\n')
            f.close()
        else:
            L = []
            while output.parent is not None:
                L.append(output.location)
                output = output.parent

            L.append(output.location)
            L.reverse()
            f.write(' '.join(L))
            f.write('\n')
            f.close()
            #print("%.3f sec.\n" % (time.time() - exec_time))

    if method == 'A*':
        #cProfile.run('A_star(locations=locations, paths=paths)')
        output = A_star(locations=locations, paths=paths)
        f = open('output.txt', 'w')
        if isinstance(output, list):
            f.write(' '.join(output))
            f.write('\n')
            f.close()
        else:
            L = []
            while output.parent is not None:
                L.append(output.location)
                output = output.parent

            L.append(output.location)
            L.reverse()
            f.write(' '.join(L))
            f.write('\n')
            f.close()
            #print("%.3f sec.\n" % (time.time() - exec_time))