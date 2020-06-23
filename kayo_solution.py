#2opt

import math
import sys
import numpy as np

from common import print_solution, read_input

def calculate_total_distance(order, distance_matrix):
    """Calculate total distance traveled for given visit order"""
    idx_from = np.array(order)
    idx_to = np.array(order[1:] + [order[0]])
    distance_arr = distance_matrix[idx_from, idx_to]

    return np.sum(distance_arr)

def calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix):
    """Calculate the difference of cost by applying given 2-opt exchange"""
    n_cities = len(visit_order)
    a, b = visit_order[i], visit_order[(i + 1) % n_cities]
    c, d = visit_order[j], visit_order[(j + 1) % n_cities]

    cost_before = distance_matrix[a, b] + distance_matrix[c, d]
    cost_after = distance_matrix[a, c] + distance_matrix[b, d]
    return cost_after - cost_before

# 二つのパスの交換後の訪問順序を計算
def apply_2opt_exchange(visit_order, i, j):
    """Apply 2-opt exhanging on visit order"""

    tmp = visit_order[i + 1: j + 1]
    tmp.reverse()
    visit_order[i + 1: j + 1] = tmp

    return visit_order

def improve_with_2opt(visit_order, distance_matrix):
    """Check all 2-opt neighbors and improve the visit order"""
    n_cities = len(visit_order)
    cost_diff_best = 0.0
    i_best, j_best = None, None

    for i in range(0, n_cities - 2):
        for j in range(i + 2, n_cities):
            if i == 0 and j == n_cities - 1:
                continue

            cost_diff = calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix)

            if cost_diff < cost_diff_best:
                cost_diff_best = cost_diff
                i_best, j_best = i, j

    if cost_diff_best < 0.0:
        visit_order_new = apply_2opt_exchange(visit_order, i_best, j_best)
        return visit_order_new
    else:
        return None

def local_search(visit_order, distance_matrix, improve_func):
    """Main procedure of local search"""
    cost_total = calculate_total_distance(visit_order, distance_matrix)

    while True:
        improved = improve_func(visit_order, distance_matrix)
        if not improved:
            break

        visit_order = improved

    return visit_order

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def solve(cities):
    N = len(cities)

    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    current_city = 0
    unvisited_cities = set(range(1, N))
    solution = [current_city]

    def distance_from_current_city(to):
        return dist[current_city][to]

    while unvisited_cities:
        next_city = min(unvisited_cities, key=distance_from_current_city)
        unvisited_cities.remove(next_city)
        solution.append(next_city)
        current_city = next_city
    return solution


if __name__ == '__main__':
    N_START = 200

    assert len(sys.argv) > 1
    # solution = solve(read_input(sys.argv[1]))
    # print(solution)
    # print_solution(solution)
    cities = np.array(read_input(sys.argv[1]))

    x = cities[:, 0]
    y = cities[:, 1]
    distance_matrix = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 +
                            (y[:, np.newaxis] - y[np.newaxis, :]) ** 2)

    test_order = list(np.random.permutation(len(cities)))

    order_best = None
    score_best = sys.float_info.max

    for _ in range(N_START):
        order_random = list(np.random.permutation(len(cities)))
        order_improved = local_search(order_random, distance_matrix, improve_with_2opt)
        score = calculate_total_distance(order_improved, distance_matrix)

        if score < score_best:
            score_best = score
            order_best = order_improved
            
    with open(sys.argv[2], mode='w') as f:
        f.write('index'+'\n')
        for ans in order_best:
            f.write(str(ans)+'\n')

    # print("index")
    # for ans in order_best:
    #     print(ans)
    # total_distance = calculate_total_distance(order_best, distance_matrix)
    # print('多スタート戦略で得られた経路の総移動距離 = {}'.format(total_distance))
