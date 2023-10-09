
#include "eps_greedy.h"

#include <algorithm>
#include <random>

using open_spiel::Action;
using open_spiel::State;

using std::vector;
using policies::GetOptimalAction;

namespace policies {

  Action EpsilonGreedyPolicy::action_selection (const State& state) {

    vector<Action> legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
      return open_spiel::kInvalidAction;
    }

    if (absl::Uniform(rng_, 0.0, 1.0) < epsilon) {
      // Choose a random action
      return legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
    }
    // Choose the best action
    return GetOptimalAction(qtable_pointer, state, abstraction_func);
  }

  void EpsilonGreedyPolicy::reward_update (const State& state, Action& action, double reward) {
    //Epsilon Greedy non necessita di alcuna propria struttura da aggiornare, è stateless
  }

  std::string EpsilonGreedyPolicy::toString () const {
    std::stringstream s;
    s << "EpsilonGreedy (" << epsilon << ")";
    return s.str();
  }

  void EpsilonGreedyPolicy::setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double disc_factor, double learn_rate, StateAbstractionFunction func) {
      qtable_pointer = table;
      abstraction_func = func;
  }


}
