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

    typedef std::function<Action(State&)> BestActionFunctor;

    private:

      BestActionFunctor bestActionFunctor;
      double epsilon;
      std::mt19937 rng_;

    public:

      EpsilonGreedyPolicy (BestActionFunctor func, double eps) {
        if (eps < 0 || eps > 1)
          std::invalid_argument("Ricevuto valore non valido");
        else
          epsilon = eps;
        bestActionFunctor = func;
      }

      virtual Action action_selection (State&);

      virtual void reward_update (State&, Action&, double);
  };

}

#endif
