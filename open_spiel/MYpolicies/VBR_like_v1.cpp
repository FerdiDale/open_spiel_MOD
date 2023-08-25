
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

  void printTab(absl::flat_hash_map<std::pair<std::string, Action>, std::pair<double, double>> tab) {
    for (auto& [key, value] : tab) {
      std::cout<<"STATO \n"<<key.first<<"\n AZIONE "<<key.second<<" MEDIA "<< value.first<<" N OSSERVAZIONI "<< value.second<< std::endl;
    }
  }

  Action VBRLikePolicyV1::action_selection (const State& state) {

      //printTab(tab_);

      vector<Action> legal_actions = state.LegalActions();
      vector<std::pair<Action, double>> mean_rewards;

      for (Action action : legal_actions) {
        mean_rewards.push_back({action, tab_[std::make_pair(state.ToString(), action)].first});
        // std::cout<<"ACTION "<<action<<" MEAN "<<tab_[std::make_pair(state.ToString(), action)].first<<std::endl;
      }

      int max_mean_index;
      double max_mean;
      bool all_equals = true;

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
                  all_equals = false;
              }
              else if (max_mean > mean_rewards[i].second) {
                  all_equals = false;
              }
            }

        }

        // std::cout<<state.ToString()<<std::endl<<std::endl;

        if (all_equals) {
          Action a = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];

          // std::cout<<"RANDOM, AZIONE "<<a<<std::endl;

          return a;
        }

        // std::cout<<"NON RANDOM, AZIONE " << mean_rewards[max_mean_index].first <<std::endl;

        return mean_rewards[max_mean_index].first;
      }


  }

  void VBRLikePolicyV1::reward_update (const State& state, Action& action, double reward) {
      double n_rewards = tab_[std::make_pair(state.ToString(), action)].second;
      double old_mean = tab_[std::make_pair(state.ToString(), action)].first;
      // std::cout<<" OLD MEAN" <<old_mean<< " N REWARDS "<<n_rewards<< " REWARD "<<reward<<" NEW MEAN "<<(old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0))<<std::endl;
      tab_[std::make_pair(state.ToString(), action)] = std::make_pair((old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0)), n_rewards+1.0);
  }

}
