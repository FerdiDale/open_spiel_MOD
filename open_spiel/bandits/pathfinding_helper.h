#ifndef PATHFINDING_HELPER_H
#define PATHFINDING_HELPER_H

#include <algorithm>
#include <random>
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"

#include <iostream>

namespace policies {

  std::vector<std::pair<int, int>> possible_next_positions(std::vector<std::vector<char>> maze, std::pair<int, int> curr);

  bool DFS (std::vector<std::vector<char>> maze, std::vector<std::vector<char>>* colors_p, std::pair<int, int> curr);

  std::string maze_to_string(std::vector<std::vector<char>> maze);

  bool traversable_maze_check(std::vector<std::vector<char>> maze, std::pair<int, int> source);

  std::string maze_gen (int n_rows = 5, int n_columns = 5, double wall_ratio = 0.2);

}

#endif
