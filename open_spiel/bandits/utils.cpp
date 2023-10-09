#include <algorithm>
#include <random>

#include "utils.h"

namespace policies {

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

  double average_of(std::vector<double> vec) {

    if (vec.empty())
      return 0;

    double sum = 0;
    for (int i = 0; i < vec.size(); i++)
      sum += vec[i];
    return sum/((double)vec.size());

  }

  Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double>* q_values,
    const State& state, StateAbstractionFunction func) { 

    std::vector<Action> legal_actions = state.LegalActions();
    Action optimal_action = open_spiel::kInvalidAction;

    double value = -1;
    for (const Action& action : legal_actions) {
      double q_val = (*q_values)[{func(state.ToString()), action}];
      if (q_val >= value) {
        value = q_val;
        optimal_action = action;
      }
    }
    return optimal_action;
  }

}
