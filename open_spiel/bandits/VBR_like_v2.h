#ifndef VBR_LIKE_V2
#define VBR_LIKE_V2

#include <algorithm>
#include <random>
#include <cmath>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "bandits/generic_policy.h"
#include "bandits/utils.h"


namespace policies {

  using policies::StateAbstractionFunction;

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

      StateAbstractionFunction abstraction_func;

      double get_best_action_qvalue (State& state);

    public :
      virtual Action action_selection (const State& state);

      virtual void reward_update (const State& state, Action& action, double reward);

      VBRLikePolicyV2(double gamma = 2, double alpha = 0.1, bool history_based = false);

      virtual void setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double disc_factor, double learn_rate, StateAbstractionFunction func) override;

      virtual std::string toString () const override;

  };

}

#endif
