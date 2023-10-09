#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_q_learning.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;

namespace policies {

  double standard_deviation_calc(std::vector<double> list, double mean);
  double average_of(std::vector<double> vec);

  Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double>* q_values,
    const State& state, StateAbstractionFunction func);

}

#endif
