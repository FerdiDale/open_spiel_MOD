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

using open_spiel::Action;
using open_spiel::State;

namespace policies {

  class EpsilonGreedyPolicy : virtual public GenericPolicy {

    typedef std::function<Action(const State&, void*)> BestActionFunctor;

    private:

      BestActionFunctor bestActionFunctor;
      double epsilon;
      void* parameters;
      std::mt19937 rng_;

    public:

      EpsilonGreedyPolicy (double eps, BestActionFunctor func, void* param) {
        if (eps < 0 || eps > 1)
          std::invalid_argument("Ricevuto valore non valido");

        epsilon = eps;
        bestActionFunctor = func;
        parameters = param;

      }

      virtual Action action_selection (const State&);

      virtual void reward_update (const State&, Action&, double);
  };

}

#endif
