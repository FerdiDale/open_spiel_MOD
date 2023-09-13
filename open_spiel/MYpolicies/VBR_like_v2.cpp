
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

  void printTab(absl::flat_hash_map<std::pair<std::string, Action>, std::vector<double>>  tab, const State& state) {

    /*std::cout<<"INIZIO TABELLA"<<std::endl;

    for (auto& [key, value] : tab) {
        std::cout<<"STATO \n"<<key.first<<"\n AZIONE "<<key.second<<" MEDIA "<< std::get<1>(value)<<" VARIANZA "<< std::get<2>(value)<<" N OSSERVAZIONI "<< std::get<3>(value) << std::endl;
    }*/
  }

  double standard_deviation_calc(std::vector<double> list, double mean) {
    double variance = 0;

    if (list.size() < 2)
      return 0;

    for (int i = 0; i < list.size(); i++) {
        variance+=(pow((list[i]-mean),2));
    }

    // std::cout<<"VARIANZA "<<variance<<" MEDIA "<<mean<<" TAGLIA LISTA "<<list.size()<<std::endl;

    return (sqrt(variance))/(list.size()-1);
  }

  double VBRLikePolicyV2::get_best_action_qvalue(State& state) {

    if (state.IsTerminal()) {
      return 0;
    }

    vector<Action> legal_actions = state.LegalActions();
    const auto state_str = state.ToString();

    Action best_action = legal_actions[0];
    double value = (*qvalues)[{state_str, best_action}];
    for (const Action& action : legal_actions) {
      double q_val = (*qvalues)[{state_str, action}];
      if (q_val >= value) {
        value = q_val;
        best_action = action;
      }
    }

    return value;
  }

  Action VBRLikePolicyV2::action_selection (const State& state) {

      printTab(tab_, state);

      vector<Action> legal_actions = state.LegalActions();
      vector<std::pair<Action, double>> UB_list;
      vector<std::pair<Action, double>> LB_list;


      if (legal_actions.empty())
        return open_spiel::kInvalidAction;

      for (Action action : legal_actions) {

        vector<double> observation_list = tab_[{state.ToString(), action}];

        double n_observations = observation_list.size();

        if (n_observations < 2) {
          return action;
        }

        double mean = (*qvalues)[{state.ToString(), action}];
        double standard_deviation = standard_deviation_calc(observation_list, mean);

        double LB = mean - (confidence_parameter*standard_deviation/sqrt(n_observations));
        double UB = mean + (confidence_parameter*standard_deviation/sqrt(n_observations));

        LB_list.push_back({action, LB});
        UB_list.push_back({action, UB});

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
        }
      }

      return acceptable_actions[absl::Uniform<int>(rng_, 0, acceptable_actions.size())];

  }

  void VBRLikePolicyV2::reward_update (const State& state, Action& action, double reward) {

      std::unique_ptr<State> next_state = state.Child(action);
      double max_next_q_value = get_best_action_qvalue(*next_state);
      double new_observation;
      if (prev_history_based)
        new_observation = (1-learning_rate) * ((*qvalues)[{state.ToString(), action}]) + learning_rate * (reward + discount_factor * max_next_q_value);
      else
        new_observation = reward + discount_factor * max_next_q_value;

    //   std::cout<<" OLD MEAN" <<old_mean<< " N REWARDS "<<n_rewards<< " REWARD "<<reward<<" NEW MEAN "<<(old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0))<<std::endl;
      tab_[{state.ToString(), action}].push_back(new_observation);
  }

  void VBRLikePolicyV2::setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double learn_rate, double disc_factor){
      qvalues = table;
      learning_rate =learn_rate;
      discount_factor = disc_factor;
  }

  VBRLikePolicyV2::VBRLikePolicyV2(double gamma, bool history_based){
    confidence_parameter = gamma;
    prev_history_based = history_based;
  }


}
