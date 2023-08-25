
#include "eps_greedy.h"

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include "generic_policy.h"

using open_spiel::Action;
using open_spiel::State;

namespace policies {

  using std::vector;

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
    return bestActionFunctor(state, parameters);
  }

  void EpsilonGreedyPolicy::reward_update (const State& state, Action& action, double reward) {
    //Epsilon Greedy non necessita di alcuna propria struttura da aggiornare, è stateless
  }

}
