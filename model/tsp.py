"""
Route sequencing using greedy nearest-neighbor + 2-opt improvement.

Tried a few approaches here — OR-Tools is overkill for 5-10 stops,
and pure brute-force blows up past 10. This hits the sweet spot:
greedy gives a decent starting point, 2-opt cleans it up.
For the problem sizes we're dealing with (5-8 stops), this is essentially optimal.
"""

import numpy as np
from model.features import haversine_km


def distance_matrix(lats, lngs):
    n = len(lats)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = haversine_km(lats[i], lngs[i], lats[j], lngs[j])
    return mat


def greedy_route(dist_mat):
    """Start from stop 0, always go to the nearest unvisited stop."""
    n = len(dist_mat)
    visited = [False] * n
    route = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = route[-1]
        nearest = min(
            (dist_mat[current][j], j)
            for j in range(n) if not visited[j]
        )[1]
        route.append(nearest)
        visited[nearest] = True

    return route


def route_length(route, dist_mat):
    return sum(dist_mat[route[i]][route[i+1]] for i in range(len(route) - 1))


def two_opt(route, dist_mat, max_rounds=100):
    """
    2-opt: try reversing every sub-segment, keep it if it shortens the route.
    Keeps going until no improvement is found or we hit max_rounds.
    Usually converges in < 10 rounds for the sizes we care about.
    """
    best = route[:]
    best_len = route_length(best, dist_mat)
    improved = True
    rounds = 0

    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                clen = route_length(candidate, dist_mat)
                if clen < best_len - 1e-10:
                    best = candidate
                    best_len = clen
                    improved = True

    return best, best_len


def solve(location_ids, lats, lngs):
    """
    Main entry point. Takes parallel lists of ids + coords,
    returns the stop ids in optimized order + total km.
    """
    if len(location_ids) <= 1:
        return location_ids, 0.0

    if len(location_ids) == 2:
        dist = haversine_km(lats[0], lngs[0], lats[1], lngs[1])
        return location_ids, round(dist, 2)

    mat = distance_matrix(lats, lngs)
    initial = greedy_route(mat)
    optimized, total_km = two_opt(initial, mat)

    ordered_ids = [location_ids[i] for i in optimized]
    return ordered_ids, round(total_km, 2)


def estimate_drive_time(total_km, avg_speed_kmh=30):
    """
    Rough ETA in hours. 30 km/h is a reasonable city average
    once you factor in signals, turns, and parking.
    """
    return round(total_km / avg_speed_kmh, 2)
