#ifndef EPS_GREEDY
#define EPS_GREEDY

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include "generic_policy.h"
#include "utils.h"

using open_spiel::Action;
using open_spiel::State;

namespace policies {

  class EpsilonGreedyPolicy : virtual public GenericPolicy {

    private:

      double epsilon;
      absl::flat_hash_map<std::pair<std::string, Action>, double>* qtable_pointer;
      StateAbstractionFunction abstraction_func;

      std::random_device rd;
      std::mt19937 rng_{rd()};

    public:

      EpsilonGreedyPolicy (double eps) {
        if (eps < 0 || eps > 1)
          std::invalid_argument("Ricevuto valore non valido");

        epsilon = eps;

      }

      virtual Action action_selection (const State&);

      virtual void reward_update (const State&, Action&, double);

      virtual void setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double disc_factor, double learn_rate, StateAbstractionFunction func) override;

      virtual std::string toString () const override;

  };

}

#endif
