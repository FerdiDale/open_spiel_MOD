#ifndef VBR_LIKE_V1
#define VBR_LIKE_V1

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "generic_policy.h"

namespace policies {

  class VBRLikePolicyV1 : public GenericPolicy {
    public :
      virtual Action& action_selection (State& state);

      virtual void reward_update (State& state, Action& action, double reward);
  };

}

#endif
