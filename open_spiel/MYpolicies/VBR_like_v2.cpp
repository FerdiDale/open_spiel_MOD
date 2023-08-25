
#include "MYpolicies/VBR_like_v2.h"

#include <algorithm>
#include <random>
#include <cmath>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "MYpolicies/generic_policy.h"

using std::vector;

namespace policies {

  void printTab(absl::flat_hash_map<std::pair<std::string, Action>, std::tuple<std::vector<double>, double, double, double>>  tab) {
    for (auto& [key, value] : tab) {
      std::cout<<"STATO \n"<<key.first<<"\n AZIONE "<<key.second<<" MEDIA "<< std::get<1>(value)<<" VARIANZA "<< std::get<2>(value)<<" N OSSERVAZIONI "<< std::get<3>(value) << std::endl;
    }
  }

  double standard_deviation_calc(std::vector<double> list, double mean) {
    double variance = 0;

    if (list.size() < 2)
      return 0;

    for (int i = 0; i < list.size(); i++) {
        variance+=(pow((list[i]-mean),2));
    }

    std::cout<<"VARIANZA "<<variance<<" MEDIA "<<mean<<" TAGLIA LISTA "<<list.size()<<std::endl;

    return (sqrt(variance))/(list.size()-1);
  }

  Action VBRLikePolicyV2::action_selection (const State& state) {

      printTab(tab_);

      vector<Action> legal_actions = state.LegalActions();
      vector<std::pair<Action, double>> UB_list;
      vector<std::pair<Action, double>> LB_list;


      if (legal_actions.empty())
        return open_spiel::kInvalidAction;

      for (Action action : legal_actions) {
        double mean = get<1>(tab_[std::make_pair(state.ToString(), action)]);
        double standard_deviation = get<2>(tab_[std::make_pair(state.ToString(), action)]);
        double n_observations = get<3>(tab_[std::make_pair(state.ToString(), action)]);

        if (n_observations < 2)
          return action;

        double LB = mean - (confidence_parameter*standard_deviation/sqrt(n_observations));
        double UB = mean + (confidence_parameter*standard_deviation/sqrt(n_observations));

        if (n_observations == 0) {
          LB = 0;
          UB = 0;
        }

        LB_list.push_back({action, LB});
        UB_list.push_back({action, UB});

        // std::cout<<"ACTION "<<action<<" MEAN "<<tab_[std::make_pair(state.ToString(), action)].first<<std::endl;
      }

      double maxLB = 0;

      for (int i = 0; i < LB_list.size(); i++) {
          if (i == 0) {
            maxLB = LB_list[0].second;
          }
          else {
            if (maxLB < LB_list[i].second) {
                maxLB = LB_list[i].second;
            }
          }

      }

      std::vector<Action> acceptable_actions;

      for (auto& [action, currUB] : UB_list) {
        if (currUB >= maxLB) {
          acceptable_actions.push_back(action);
          std::cout<<" UB "<<currUB<< " AZIONE "<<action<<std::endl; 
        }
      }

      if (acceptable_actions.empty())
        return legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
      else
        return acceptable_actions[absl::Uniform<int>(rng_, 0, acceptable_actions.size())];

  }

  void VBRLikePolicyV2::reward_update (const State& state, Action& action, double reward) {
      double n_rewards = std::get<3>(tab_[std::make_pair(state.ToString(), action)]);
      double old_mean = std::get<1>(tab_[std::make_pair(state.ToString(), action)]);
      auto rewards_list = std::get<0>(tab_[{state.ToString(), action}]);
    //   std::cout<<" OLD MEAN" <<old_mean<< " N REWARDS "<<n_rewards<< " REWARD "<<reward<<" NEW MEAN "<<(old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0))<<std::endl;
      double new_mean = (old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0));
      n_rewards++;
      rewards_list.push_back(reward);
      double standard_dev = standard_deviation_calc(rewards_list, new_mean);
      tab_[{state.ToString(), action}] = {rewards_list, new_mean, standard_dev, n_rewards};
  }

  VBRLikePolicyV2::VBRLikePolicyV2(double gamma){
    confidence_parameter = gamma;
  }


}
