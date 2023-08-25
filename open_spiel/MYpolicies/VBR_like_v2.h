#ifndef VBR_LIKE_V2
#define VBR_LIKE_V2

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "MYpolicies/generic_policy.h"

namespace policies {

  class VBRLikePolicyV2 : public GenericPolicy {

    private :
      absl::flat_hash_map<std::pair<std::string, Action>, std::tuple<std::vector<double>, double, double, double>> tab_;

      std::mt19937 rng_;

      double confidence_parameter; //gamma

    public :
      virtual Action action_selection (const State& state);

      virtual void reward_update (const State& state, Action& action, double reward);

      VBRLikePolicyV2(double gamma = 2);
  };

}

#endif
