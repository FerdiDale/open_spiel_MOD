// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_
#define OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/spiel.h"
#include "bandits/generic_policy.h"

#include <random>
#include <iostream>

using policies::GenericPolicy;
using policies::StateAbstractionFunction;

namespace open_spiel {
namespace algorithms {

// Tabular Q-learning algorithm: solves for the optimal action value function
// of a game.
// It considers all states with depth at most depth_limit from the
// initial state (so if depth_limit is 0, only the root is considered).
// If depth limit is negative, all states are considered.
//
// Currently works for sequential 1-player or 2-player zero-sum games.
//
// Based on the implementation in Sutton and Barto, Intro to RL. Second Edition,
// 2018. Section 6.5.
//
// Includes implementation of Watkins’s Q(lambda) which can be found in
// Sutton and Barto, Intro to RL. Second Edition, 2018. Section 12.10.
// (E.g. https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
// Eligibility traces are implemented with the "accumulate"
// method (+1 at each iteration) instead of "replace" implementation
// (doesn't sum trace values). Parameter lambda_ determines the level
// of bootstraping.

std::string identity_function (const std::string state);

class TabularQLearningSolver {

  static inline constexpr double kDefaultDepthLimit = -1;
  static inline constexpr double kDefaultEpsilon = 0.01;
  static inline constexpr double kDefaultLearningRate = 0.01;
  static inline constexpr double kDefaultDiscountFactor = 0.99;
  static inline constexpr double kDefaultLambda = 0;

 public:
  TabularQLearningSolver(std::shared_ptr<const Game> game, StateAbstractionFunction func = identity_function);

  TabularQLearningSolver(std::shared_ptr<const Game> game, double depth_limit,
                         double epsilon, double learning_rate,
                         double discount_factor, double lambda, StateAbstractionFunction func = identity_function);

  TabularQLearningSolver(std::shared_ptr<const Game> game, GenericPolicy* policy, StateAbstractionFunction func = identity_function);

  TabularQLearningSolver(
    std::shared_ptr<const Game> game, double learning_rate, double discount_factor, GenericPolicy* policy, StateAbstractionFunction func);

  void RunIteration();

  const absl::flat_hash_map<std::pair<std::string, Action>, double>&
  GetQValueTable() const;
  StateAbstractionFunction GetAbstractionFunction() const;

 private:
  // Given a player and a state, gets the best possible action from this state
  Action GetBestAction(const State& state, double min_utility);

  Action GetBestAction(const State& state);

  // Given a state, gets the best possible action value from this state
  double GetBestActionValue(const State& state, double min_utility);

  // Given a player and a state, gets the action, sampled from an epsilon-greedy
  // policy. Returns <action, chosen_uniformly> where the second element
  // indicates whether an action was chosen uniformly (which occurs with epsilon
  // chance).
  std::pair<Action, bool> SampleActionFromEpsilonGreedyPolicy(
      const State& state, double min_utility);

  // Moves a chance node to the next decision/terminal node by sampling from
  // the legal actions repeatedly
  void SampleUntilNextStateOrTerminal(State* state);

  std::shared_ptr<const Game> game_;
  int depth_limit_;
  double epsilon_;
  double learning_rate_;
  double discount_factor_;
  double lambda_;
  std::mt19937 rng_{(std::random_device())()};
  GenericPolicy* policy_;
  std::tuple<absl::flat_hash_map<std::pair<std::string, Action>, double>*, double, StateAbstractionFunction> tuple;
  absl::flat_hash_map<std::pair<std::string, Action>, double> values_;
  absl::flat_hash_map<std::pair<std::string, Action>, double>
      eligibility_traces_;
  StateAbstractionFunction abstraction_func;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_TABULAR_Q_LEARNING_H_
