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

#include "open_spiel/algorithms/tabular_q_learning.h"

#include <algorithm>
#include <random>
#include <typeinfo>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "MYpolicies/generic_policy.h"
#include "MYpolicies/eps_greedy.h"
#include "MYpolicies/VBR_like_v1.h"
#include "MYpolicies/VBR_like_v2.h"

namespace open_spiel {
namespace algorithms {

using policies::GenericPolicy;
using policies::EpsilonGreedyPolicy;
using policies::VBRLikePolicyV1;
using policies::VBRLikePolicyV2;
using std::vector;

Action getBestActionFunction (const State& state, void* data_pointer) {
  auto data_cast = *((std::pair<absl::flat_hash_map<std::pair<std::string, Action>, double>*, double>*)data_pointer);
  auto values_ = data_cast.first;
  auto min_utility = data_cast.second;

  vector<Action> legal_actions = state.LegalActions();
  const auto state_str = state.ToString();

  Action best_action = legal_actions[0];
  double value = min_utility;
  for (const Action& action : legal_actions) {
    double q_val = (*values_)[{state_str, action}];
    if (q_val >= value) {
      value = q_val;
      best_action = action;
    }
  }
  return best_action;
}

Action TabularQLearningSolver::GetBestAction(const State& state,
                                             double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  const auto state_str = state.ToString();

  Action best_action = legal_actions[0];
  double value = min_utility;
  for (const Action& action : legal_actions) {
    double q_val = values_[{state_str, action}];
    if (q_val >= value) {
      value = q_val;
      best_action = action;
    }
  }
  return best_action;
}

double TabularQLearningSolver::GetBestActionValue(const State& state,
                                                  double min_utility) {
  if (state.IsTerminal()) {
    // q(s,a) is 0 when s is terminal.
    return 0;
  }
  return values_[{state.ToString(), GetBestAction(state, min_utility)}];
}

std::pair<Action, bool>
TabularQLearningSolver::SampleActionFromEpsilonGreedyPolicy(
    const State& state, double min_utility) {
  return std::make_pair(policy_->action_selection(state), true);
}

void TabularQLearningSolver::SampleUntilNextStateOrTerminal(State* state) {
  // Repeatedly sample while chance node, so that we end up at a decision node
  while (state->IsChanceNode() && !state->IsTerminal()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    state->ApplyAction(SampleAction(outcomes, rng_).first);
  }
}

TabularQLearningSolver::TabularQLearningSolver(std::shared_ptr<const Game> game) : game_(game),
    depth_limit_(kDefaultDepthLimit),
    epsilon_(kDefaultEpsilon),
    learning_rate_(kDefaultLearningRate),
    discount_factor_(kDefaultDiscountFactor),
    lambda_(kDefaultLambda) {

        pair = {&values_, game_->MinUtility()};

        policy_ = new EpsilonGreedyPolicy(kDefaultEpsilon, getBestActionFunction, &pair);

        SPIEL_CHECK_LE(lambda_, 1);
        SPIEL_CHECK_GE(lambda_, 0);

        // Currently only supports 1-player or 2-player zero sum games
        SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
        if (game_->NumPlayers() == 2) {
          SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
        }

        // No support for simultaneous games (needs an LP solver). And so also must
        // be a perfect information game.
        // SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
        // SPIEL_CHECK_EQ(game_->GetType().information,
        //                GameType::Information::kPerfectInformation);

}

// TabularQLearningSolver::TabularQLearningSolver(std::shared_ptr<const Game> game)
//     : TabularQLearningSolver(game, new VBRLikePolicyV2()) {
// }

TabularQLearningSolver::TabularQLearningSolver(
    std::shared_ptr<const Game> game, double depth_limit, double epsilon,
    double learning_rate, double discount_factor, double lambda)  : game_(game),
      depth_limit_(depth_limit),
      epsilon_(epsilon),
      learning_rate_(learning_rate),
      discount_factor_(discount_factor),
      lambda_(lambda){

          pair = {&values_, game_->MinUtility()};

          SPIEL_CHECK_LE(lambda_, 1);
          SPIEL_CHECK_GE(lambda_, 0);

          // Currently only supports 1-player or 2-player zero sum games
          SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
          if (game_->NumPlayers() == 2) {
            SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
          }


        // No support for simultaneous games (needs an LP solver). And so also must
        // be a perfect information game.
        // SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
        // SPIEL_CHECK_EQ(game_->GetType().information,
                      //  GameType::Information::kPerfectInformation);

        policy_ = new EpsilonGreedyPolicy(kDefaultEpsilon, getBestActionFunction, &pair);
  }

  TabularQLearningSolver::TabularQLearningSolver(std::shared_ptr<const Game> game, GenericPolicy* policy) : game_(game),
    depth_limit_(kDefaultDepthLimit),
    epsilon_(kDefaultEpsilon),
    learning_rate_(kDefaultLearningRate),
    discount_factor_(kDefaultDiscountFactor),
    lambda_(kDefaultLambda),
    policy_(policy) {

        pair = {&values_, game_->MinUtility()};

        if (typeid(*policy_) == typeid(VBRLikePolicyV2)) {
          ((VBRLikePolicyV2*)policy_)->setQTableStructure(&values_, learning_rate_, discount_factor_);
        }

        SPIEL_CHECK_LE(lambda_, 1);
        SPIEL_CHECK_GE(lambda_, 0);

        // Currently only supports 1-player or 2-player zero sum games
        SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
        if (game_->NumPlayers() == 2) {
          SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
        }

        // No support for simultaneous games (needs an LP solver). And so also must
        // be a perfect information game.
        // SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
        // SPIEL_CHECK_EQ(game_->GetType().information,
                      //  GameType::Information::kPerfectInformation);

}

const absl::flat_hash_map<std::pair<std::string, Action>, double>&
TabularQLearningSolver::GetQValueTable() const {
  // std::cout<<"PRE TRASFERIMENTO"<<std::endl;
  // for (const auto&[k,v] : values_) {
  //   std::cout<<"STATO "<<k.first<<" AZIONE "<<k.second<<" QVALUE "<<v<<std::endl;
  // }
  return values_;
}

void TabularQLearningSolver::RunIteration() {
  const double min_utility = game_->MinUtility();
  // Choose start state
  std::unique_ptr<State> curr_state = game_->NewInitialState();
  SampleUntilNextStateOrTerminal(curr_state.get());

  while (!curr_state->IsTerminal()) {
    const Player player = curr_state->CurrentPlayer();
    std::cout<<"PLAYER "<<player<<std::endl;

    // Sample action from the state using an epsilon-greedy policy
    auto [curr_action, chosen_uniformly] =
        SampleActionFromEpsilonGreedyPolicy(*curr_state, min_utility);

    std::unique_ptr<State> next_state = curr_state->Child(curr_action);
    SampleUntilNextStateOrTerminal(next_state.get());

    const double reward = next_state->Rewards()[player];
    // Next q-value in perspective of player to play at curr_state (important
    // note: exploits property of two-player zero-sum)
    const double next_q_value =
        (player != next_state->CurrentPlayer() ? -1 : 1) *
        GetBestActionValue(*next_state, min_utility);

    // Update the q value
    std::string key = curr_state->ToString();
    double new_q_value = reward + discount_factor_ * next_q_value;
    std::cout<<"REWARD "<<reward<<" NEXT_Q_VALUE "<<next_q_value<<std::endl;

    double prev_q_val = values_[{key, curr_action}];
    if (lambda_ == 0) {
      // If lambda_ is equal to zero run Q-learning as usual.
      // It's not necessary to update eligibility traces.
      values_[{key, curr_action}] +=
          learning_rate_ * (new_q_value - prev_q_val);
      std::cout<<"STATO "<<key<<" AZIONE "<<curr_action<<" QVALUE "<<values_[{key, curr_action}]<<std::endl;
    } else {
      double lambda =
          player != next_state->CurrentPlayer() ? -lambda_ : lambda_;
      eligibility_traces_[{key, curr_action}] += 1;

      for (const auto& q_cell : values_) {
        std::string state = q_cell.first.first;
        Action action = q_cell.first.second;

        values_[{state, action}] += learning_rate_ *
                                    (new_q_value - prev_q_val) *
                                    eligibility_traces_[{state, action}];
        if (chosen_uniformly) {
          eligibility_traces_[{state, action}] = 0;
        } else {
          eligibility_traces_[{state, action}] *= discount_factor_ * lambda;
        }
      }
    }

    policy_->reward_update(*(curr_state.get()), curr_action, reward);

    curr_state = std::move(next_state);
  }
}
}  // namespace algorithms
}  // namespace open_spiel


//CODICE ORIGINALE

/*#include "open_spiel/algorithms/tabular_q_learning.h"

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

using std::vector;

Action TabularQLearningSolver::GetBestAction(const State& state,
                                             double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  const auto state_str = state.ToString();

  Action best_action = legal_actions[0];
  double value = min_utility;
  for (const Action& action : legal_actions) {
    double q_val = values_[{state_str, action}];
    if (q_val >= value) {
      value = q_val;
      best_action = action;
    }
  }
  return best_action;
}

double TabularQLearningSolver::GetBestActionValue(const State& state,
                                                  double min_utility) {
  if (state.IsTerminal()) {
    // q(s,a) is 0 when s is terminal.
    return 0;
  }
  return values_[{state.ToString(), GetBestAction(state, min_utility)}];
}

std::pair<Action, bool>
TabularQLearningSolver::SampleActionFromEpsilonGreedyPolicy(
    const State& state, double min_utility) {
  vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.empty()) {
    return {kInvalidAction, false};
  }

  if (absl::Uniform(rng_, 0.0, 1.0) < epsilon_) {
    // Choose a random action
    return {legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())],
            true};
  }
  // Choose the best action
  return {GetBestAction(state, min_utility), false};
}

void TabularQLearningSolver::SampleUntilNextStateOrTerminal(State* state) {
  // Repeatedly sample while chance node, so that we end up at a decision node
  while (state->IsChanceNode() && !state->IsTerminal()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    state->ApplyAction(SampleAction(outcomes, rng_).first);
  }
}

TabularQLearningSolver::TabularQLearningSolver(std::shared_ptr<const Game> game)
    : game_(game),
      depth_limit_(kDefaultDepthLimit),
      epsilon_(kDefaultEpsilon),
      learning_rate_(kDefaultLearningRate),
      discount_factor_(kDefaultDiscountFactor),
      lambda_(kDefaultLambda) {
  SPIEL_CHECK_LE(lambda_, 1);
  SPIEL_CHECK_GE(lambda_, 0);

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
  if (game_->NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_->GetType().information,
                 GameType::Information::kPerfectInformation);
}

TabularQLearningSolver::TabularQLearningSolver(
    std::shared_ptr<const Game> game, double depth_limit, double epsilon,
    double learning_rate, double discount_factor, double lambda)
    : game_(game),
      depth_limit_(depth_limit),
      epsilon_(epsilon),
      learning_rate_(learning_rate),
      discount_factor_(discount_factor),
      lambda_(lambda) {
  SPIEL_CHECK_LE(lambda_, 1);
  SPIEL_CHECK_GE(lambda_, 0);

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game_->NumPlayers() == 1 || game_->NumPlayers() == 2);
  if (game_->NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game_->GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_->GetType().information,
                 GameType::Information::kPerfectInformation);
}

const absl::flat_hash_map<std::pair<std::string, Action>, double>&
TabularQLearningSolver::GetQValueTable() const {
  return values_;
}

void TabularQLearningSolver::RunIteration() {
  const double min_utility = game_->MinUtility();
  // Choose start state
  std::unique_ptr<State> curr_state = game_->NewInitialState();
  SampleUntilNextStateOrTerminal(curr_state.get());

  while (!curr_state->IsTerminal()) {
    const Player player = curr_state->CurrentPlayer();

    // Sample action from the state using an epsilon-greedy policy
    auto [curr_action, chosen_uniformly] =
        SampleActionFromEpsilonGreedyPolicy(*curr_state, min_utility);

    std::unique_ptr<State> next_state = curr_state->Child(curr_action);
    SampleUntilNextStateOrTerminal(next_state.get());

    const double reward = next_state->Rewards()[player];
    // Next q-value in perspective of player to play at curr_state (important
    // note: exploits property of two-player zero-sum)
    const double next_q_value =
        (player != next_state->CurrentPlayer() ? -1 : 1) *
        GetBestActionValue(*next_state, min_utility);

    // Update the q value
    std::string key = curr_state->ToString();
    double new_q_value = reward + discount_factor_ * next_q_value;

    double prev_q_val = values_[{key, curr_action}];
    if (lambda_ == 0) {
      // If lambda_ is equal to zero run Q-learning as usual.
      // It's not necessary to update eligibility traces.
      values_[{key, curr_action}] +=
          learning_rate_ * (new_q_value - prev_q_val);
    } else {
      double lambda =
          player != next_state->CurrentPlayer() ? -lambda_ : lambda_;
      eligibility_traces_[{key, curr_action}] += 1;

      for (const auto& q_cell : values_) {
        std::string state = q_cell.first.first;
        Action action = q_cell.first.second;

        values_[{state, action}] += learning_rate_ *
                                    (new_q_value - prev_q_val) *
                                    eligibility_traces_[{state, action}];
        if (chosen_uniformly) {
          eligibility_traces_[{state, action}] = 0;
        } else {
          eligibility_traces_[{state, action}] *= discount_factor_ * lambda;
        }
      }
    }

    curr_state = std::move(next_state);
  }
}
}  // namespace algorithms
}  // namespace open_spiel*/
