#ifndef GENERIC_POLICY_H
#define GENERIC_POLICY_H

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"

using open_spiel::Action;
using open_spiel::State;

namespace policies {

  typedef std::function<std::string(const std::string)> StateAbstractionFunction;

  class GenericPolicy {
    public :
      virtual Action action_selection (const State& state) = 0;

      virtual void reward_update (const State& state, Action& action, double reward) = 0;

      virtual void setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double disc_factor, double learn_rate, policies::StateAbstractionFunction func) {};

      virtual std::string toString () const = 0;
      
  };

}

#endif
