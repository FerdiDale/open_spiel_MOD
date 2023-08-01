
#include "MYpolicies/VBR_like_v1.h"

#include <algorithm>
#include <random>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "MYpolicies/generic_policy.h"

using std::vector;

namespace policies {

  Action VBRLikePolicyV1::action_selection (const State& state) {

      vector<Action> legal_actions = state.LegalActions();
      vector<std::pair<Action, double>> mean_rewards;

      for (Action action : legal_actions) {
        mean_rewards.push_back({action, tab_[std::make_pair(state.ToString(), action)].first});
        std::cout<<"ACTION "<<action<<" MEAN "<<tab_[std::make_pair(state.ToString(), action)].first<<std::endl;
      }

      int max_mean_index;
      double max_mean;

      if (legal_actions.empty())
        return open_spiel::kInvalidAction;
      else {
        for (int i = 0; i < mean_rewards.size(); i++) {
            if (i == 0) {
              max_mean_index = 0;
              max_mean = mean_rewards[0].second;
            }
            else {
              if (max_mean < mean_rewards[i].second) {
                  max_mean_index = i;
                  max_mean = mean_rewards[i].second;
              }
            }

        }

        std::cout<<"RANDOM"<<std::endl;

        if (max_mean==0) {
          Action a = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
          return a;
        }

        std::cout<<"NON RANDOM, AZIONE " << mean_rewards[max_mean_index].first <<std::endl;

        return mean_rewards[max_mean_index].first;
      }


  }

  void VBRLikePolicyV1::reward_update (const State& state, Action& action, double reward) {
      double n_rewards = tab_[std::make_pair(state.ToString(), action)].second;
      double old_mean = tab_[std::make_pair(state.ToString(), action)].first;
      tab_[std::make_pair(state.ToString(), action)] = std::make_pair((old_mean*n_rewards/(n_rewards+1))+(reward/(n_rewards+1)), n_rewards+1);
  }

}
