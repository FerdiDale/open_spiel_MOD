
#include "VBR_like_v4.h"

using std::vector;
using policies::standard_deviation_calc;

namespace policies {

  double VBRLikePolicyV4::get_best_action_qvalue(State& state) {

    if (state.IsTerminal()) {
      return 0;
    }

    vector<Action> legal_actions = state.LegalActions();
    const auto state_str = abstraction_func(state.ToString());

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

  Action VBRLikePolicyV4::action_selection (const State& state) {

      vector<Action> legal_actions = state.LegalActions();
      absl::flat_hash_map<Action, double> sample;

      if (legal_actions.empty())
        return open_spiel::kInvalidAction;

      for (Action action : legal_actions) {

        vector<double> observation_list = tab_[{abstraction_func(state.ToString()), action}];

        double n_observations = observation_list.size();

        if (n_observations < 2) {
          return action;
        }

        double mean = (*qvalues)[{abstraction_func(state.ToString()), action}];
        double standard_deviation = standard_deviation_calc(observation_list, mean);

        double standard_error = standard_deviation/sqrt(n_observations);

        std::normal_distribution<double> dist (mean, standard_error); //NB: prende in input media e deviazione standard, non varianza

        sample[action] = dist(rng_);

      }

      Action max_action = legal_actions[0];
      double max_value = sample[max_action];

      for (auto& [k,v] : sample) {
        if (v > max_value) {
          max_action = k;
          max_value = v;
        }
      }

      return max_action;

  }

  void VBRLikePolicyV4::reward_update (const State& state, Action& action, double reward) {

      std::unique_ptr<State> next_state = state.Child(action);
      double max_next_q_value = get_best_action_qvalue(*next_state);
      double new_observation;
      if (prev_history_based)
        new_observation = (1-learning_rate) * ((*qvalues)[{abstraction_func(state.ToString()), action}]) + learning_rate * (reward + discount_factor * max_next_q_value);
      else
        new_observation = reward + discount_factor * max_next_q_value;

    //   std::cout<<" OLD MEAN" <<old_mean<< " N REWARDS "<<n_rewards<< " REWARD "<<reward<<" NEW MEAN "<<(old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0))<<std::endl;
      tab_[{abstraction_func(state.ToString()), action}].push_back(new_observation);
  }

  void VBRLikePolicyV4::setQTableStructure(absl::flat_hash_map<std::pair<std::string, Action>, double>* table, double disc_factor, double learn_rate, StateAbstractionFunction func){
      qvalues = table;
      discount_factor = disc_factor;
      abstraction_func = func;
  }

  VBRLikePolicyV4::VBRLikePolicyV4(double gamma, double alpha, bool history_based){
    confidence_parameter = gamma;
    prev_history_based = history_based;
    alpha = learning_rate;
  }

  std::string VBRLikePolicyV4::toString () const {
    std::stringstream s;
    s << "VBRLike4 (" << (prev_history_based ? "history" : "no history") << ") alfa(" << learning_rate << ")";
    return s.str();
  }



}
