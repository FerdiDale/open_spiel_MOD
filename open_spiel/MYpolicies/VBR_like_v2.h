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
      absl::flat_hash_map<std::pair<std::string, Action>, std::vector<double>> tab_;

      std::random_device rd;
      std::mt19937 rng_{rd()};

      double confidence_parameter; //gamma

      double learning_rate; //Q-learning alpha
      double discount_factor; //Q-learning gamma

      absl::flat_hash_map<std::pair<std::string, Action>, double>* qvalues = nullptr;

      bool prev_history_based;

      double get_best_action_qvalue (State& state);

    public :
      virtual Action action_selection (const State& state);

      virtual void reward_update (const State& state, Action& action, double reward);

      VBRLikePolicyV2(double gamma = 2, bool history_based = false);

      void setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double learn_rate, double disc_factor);
  };

}

#endif
