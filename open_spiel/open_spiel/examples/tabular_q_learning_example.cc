// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_q_learning.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/pathfinding.h"
#include "bandits/generic_policy.h"
#include "bandits/eps_greedy.h"
#include "bandits/VBR_like_v1.h"
#include "bandits/VBR_like_v2.h"
#include "bandits/VBR_Thompson_like.h"
#include "bandits/pathfinding_helper.h"
#include "bandits/state_abstraction_functions.h"

#include <iostream>
#include <fstream>

using policies::GenericPolicy;
using policies::EpsilonGreedyPolicy;
using policies::VBRLikePolicyV1;
using policies::VBRLikePolicyV2;
using policies::VBRThompsonLikePolicy;

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::TurnBasedSimultaneousGame;
using open_spiel::LoadGameAsTurnBased;
using open_spiel::GameParameters;
using open_spiel::GameParameter;
using open_spiel::GameType;

using open_spiel::algorithms::TabularQLearningSolver;
using open_spiel::pathfinding::PathfindingGame;
using policies::StateAbstractionFunction;

using policies::standard_deviation_calc;
using policies::average_of;
using policies::GetOptimalAction;

using policies::maze_gen;
using policies::BFS;

using policies::identity;
using policies::visibility_limit_no_distinction;
using policies::visibility_limit_with_distinction;

struct test_parameters {
  int n_reps = 10;
  int n_phases = 10;
  int n_training = 10000;
  int n_playing = 1000;
  StateAbstractionFunction abstraction_func = identity;
  std::string tag = "id"; //Tag stringa per aggiungere informazioni per identificare il test in base alle sue caratteristiche
  
};

struct qlearning_parameters {
  double learning_rate = 0.01;
  double discount_factor = 0.99;
};

struct pathfinding_parameters {
  int horizon = 100;
  int n_rows = 5;
  int n_columns = 5;
  double wall_ratio = 0.2;
  double random_move_chance = 0;
  int maze_repetitions = 1;
};

GameParameters PFParametersToGameParameters(pathfinding_parameters params) {
  GameParameters gparams;
  gparams["random_move_chance"] = GameParameter(params.random_move_chance);
  std::string maze = maze_gen(params.n_rows, params.n_columns, params.wall_ratio);
  gparams["grid"] = GameParameter(maze);
  gparams["horizon"] = GameParameter(params.horizon);
  return gparams;
}

absl::flat_hash_map<int, std::vector<std::pair<int, double>>> TestGenericGame
(std::shared_ptr<const Game> game, std::vector<GenericPolicy*> policy_vec, test_parameters t_parameters, qlearning_parameters q_parameters) {

  int n_reps = t_parameters.n_reps;
  int n_phases = t_parameters.n_phases;
  int n_training = t_parameters.n_training;
  int n_playing = t_parameters.n_playing;
  StateAbstractionFunction abstraction_func = t_parameters.abstraction_func;
  std::string tag = t_parameters.tag;

  double learning_rate = q_parameters.learning_rate;
  double discount_factor = q_parameters.discount_factor;

  GameParameters game_parameters;

  if (game->GetParameters()["game"].has_game_value()) {
    game_parameters = game->GetParameters()["game"].game_value();
  } else {
    game_parameters = game->GetParameters();
  }

  std::string game_name;

  if (game_parameters["name"].has_string_value()) {
    game_name = game_parameters["name"].string_value();
  } else {
    game_name = game->GetType().short_name;
  }

  std::vector<TabularQLearningSolver*> vec_algos;
  for (GenericPolicy* policy : policy_vec) {
    TabularQLearningSolver* qlearning_algo = new TabularQLearningSolver(game, learning_rate, discount_factor, policy, abstraction_func);
    vec_algos.push_back(qlearning_algo);
  }

  std::random_device rd;
  std::mt19937 rng_(rd());

  double n_wins;
  double win_percentage;
  absl::flat_hash_map<int, std::vector<std::pair<int, double>>> phase_scores; //Usiamo un identificativo intero per riconoscere gli algoritmi, corrisponderanno alla loro posizione in vec_algos

  std::cout<<"INIZIO INTERNO"<<std::endl;
  for (int algo_id = 0; algo_id < vec_algos.size();  algo_id++) {
    for (int phase = 0; phase < n_phases; phase++) { //Ripetiamo il test in n_phases fasi per notare l'evoluzione dei risultati al miglioramento della tabella

      for (int iter = 0; iter < n_training; iter++) { //Eseguiamo n_training iterazioni in cui addestriamo l'agente
        // std::cout<<"FASE NUMERO "<<phase+1<<" ITERAZIONE NUMERO "<<iter+1<<std::endl;
        vec_algos[algo_id]->RunIteration();
      }

      n_wins = 0;
      std::vector<double> vec_returns;
      absl::flat_hash_map<std::pair<std::string, Action>, double> QTable = vec_algos[algo_id]->GetQValueTable();

      for (int match = 0; match < n_playing; match++) { //L'agente gioca al suo meglio n_playing volte, per avere una stima accurata della sua bravura
        std::unique_ptr<State> state = game->NewInitialState();
        while (!state->IsTerminal()) {
          if (state->CurrentPlayer() != 0) {
            std::vector<Action> legal_actions = state->LegalActions();
            Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
            state->ApplyAction(random_action);
          }
          else {
            Action optimal_action = GetOptimalAction(&QTable, *state, abstraction_func);
            state->ApplyAction(optimal_action);
          }
        }
        vec_returns.push_back(state->Returns()[0]);
      }

      if (game_name == "pathfinding") { //Dobbiamo ricavare il numero di passi impiegato partendo dal valore ritornato, lavoriamo diversamente

        int horizon = game_parameters["horizon"].int_value();
        double success_reward = game_parameters["solve_reward"].double_value()+game_parameters["group_reward"].double_value();
        double penalty = abs(game_parameters["step_reward"].double_value());
        std::string grid = game_parameters["grid"].string_value();
        int minpassi = BFS(grid);

        std::vector<double> vec_passi;

        for (double ret : vec_returns) {
          if (ret - horizon * penalty < 0.1) { //L'uguaglianza tra double si comporta in modo inconsistente
            vec_passi.push_back(horizon);
          }
          else {
            vec_passi.push_back(((success_reward-ret)/penalty)+1.0);
          }
        }

        double avg = 0;
        for (double n_passi : vec_passi) {
          double relative_value = (horizon-n_passi)/((double)(horizon-minpassi));
          avg+=relative_value;
        }
        avg/=vec_passi.size();
        phase_scores[algo_id].push_back({phase+1, avg});
      }
      else {
        for (double ret : vec_returns) {
          // if (ret == game->MaxUtility()) //Vittoria stretta, bisogna controllare in base al singolo gioco se funziona
          //   n_wins++;
          if (ret >= 0) //Vittoria o pareggio, solitamente funziona ma conviene comunque controllare i ritorni del singolo gioco
            n_wins++;
        }
        win_percentage = n_wins/((double)n_playing);
        phase_scores[algo_id].push_back({phase+1, win_percentage});
      }

    }

  }

  std::cout<<"FINE INTERNO"<<std::endl;

  for (TabularQLearningSolver* qlearning : vec_algos) {
    delete qlearning;
  }

  return phase_scores;

}

void TestGenericGameMulti(std::string game_name, std::vector<GenericPolicy*> policy_vec, test_parameters t_parameters, qlearning_parameters q_parameters, 
  pathfinding_parameters p_parameters = {}) {

  std::random_device rd;
  std::mt19937 rng_(rd());

  absl::flat_hash_map<int, absl::flat_hash_map<int, std::vector<double>>>* results = new absl::flat_hash_map<int, absl::flat_hash_map<int, std::vector<double>>>; //A ogni algoritmo sono associate n_phases fasi, ad ogni fase sono associate n_reps risultati

  double baseline_wins = 0;

  int n_reps = t_parameters.n_reps;
  int n_phases = t_parameters.n_phases;
  int n_training = t_parameters.n_training;
  int n_playing = t_parameters.n_playing;
  StateAbstractionFunction abstraction_func = t_parameters.abstraction_func;
  std::string tag = t_parameters.tag;

  int horizon = p_parameters.horizon;
  int n_rows = p_parameters.n_rows;
  int n_columns = p_parameters.n_columns;
  double wall_ratio = p_parameters.wall_ratio;
  double random_move_chance = p_parameters.random_move_chance;
  int maze_reps = p_parameters.maze_repetitions;

  GameParameters game_parameters;

  for (int rep = 0; rep < n_reps; rep++) {

    GameParameters setting_parameters;
    if (game_name == "pathfinding") {
      setting_parameters = PFParametersToGameParameters(p_parameters);
    }

    std::shared_ptr<const Game> game_pointer = LoadGameAsTurnBased(game_name, setting_parameters);

    GameParameters game_parameters;

    if (game_pointer->GetParameters()["game"].has_game_value()) {
    game_parameters = game_pointer->GetParameters()["game"].game_value();
    } else {
      game_parameters = game_pointer->GetParameters();
    }

    //Simuliamo un gioco n_playing*n_reps*n_phases volte in cui si gioca totalmente a caso, per generare una baseline dei risultati, utilizzeremo poi la media dei reward.
    int n_random_matches = n_playing*n_phases;
    n_random_matches*=maze_reps; //Se il gioco è pathfinding dobbiamo tenere conto delle volte che si ripete il singolo labirinto (Se non è pathfinding varrà 1)

    for (int match = 0; match < n_random_matches; match++) {
      std::unique_ptr<State> state = game_pointer->NewInitialState();
      while (!state->IsTerminal()) {
        std::vector<Action> legal_actions = state->LegalActions();
        Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
        state->ApplyAction(random_action);
      }
      if (game_name == "pathfinding") {

        int horizon = game_parameters["horizon"].int_value();
        double success_reward = game_parameters["solve_reward"].double_value()+game_parameters["group_reward"].double_value();
        double penalty = abs(game_parameters["step_reward"].double_value());
        std::string grid = game_parameters["grid"].string_value();
        int minpassi = BFS(grid);
        int n_passi;

        if (state->Returns()[0] - horizon * penalty < 0.1) { //L'uguaglianza tra double si comporta in modo inconsistente
          n_passi = horizon;
        }
        else {
          n_passi = (((success_reward-state->Returns()[0])/penalty)+1);
        }

        baseline_wins+=((horizon-n_passi)/((double)(horizon-minpassi)));
      }
      else {
        if (state->Returns()[0] >= 0) //Se il gioco non è pathfinding assumiamo che ci basti controllare se il gioco ritorna un valore nonnegativo come vittoria/pareggio
          baseline_wins++;
      }
    }

    //Testiamo effettivamente il gioco, utilizzando i risultati accumulati dell'iterazione

    for (int m_rep = 0; m_rep < maze_reps; m_rep++) { //Ripetiamo lo stesso test maze_repetitions volte

      std::cout<<"LABIRINTO NUMERO "<<rep<<" RIPETIZIONE NUMERO "<<m_rep<<std::endl;
     
      absl::flat_hash_map<int, std::vector<std::pair<int, double>>> curr_res = TestGenericGame(game_pointer, policy_vec, t_parameters, q_parameters);

      std::cout<<"FINE TEST"<<std::endl<<std::endl;

      for (int algo = 0; algo < curr_res.size(); algo++) { //La taglia della mappa sarà data dal numero di algoritmi utilizzati nel test
        std::vector<std::pair<int, double>> phase_scores_algo = curr_res.at(algo);
        for (int i = 0; i < phase_scores_algo.size(); i++) {
            std::pair<int, double> curr_pair = phase_scores_algo[i];
            (*results)[algo][curr_pair.first].push_back(curr_pair.second); //Aggiungiamo alla coppia algoritmo-fase il valore ricavato nella ripetizione rep (corrente)
        }
      }

    }

  }

  double baseline_value = baseline_wins/((double)(n_playing*n_reps*n_phases*maze_reps));

  double max_y;
  double min_y;

  for (int algo = 0; algo < results->size(); algo++) {
  absl::flat_hash_map<int, std::vector<double>>& map = results->at(algo);
    for (int phase = 1; phase <= map.size(); phase++) { //Le fasi sono numerate da 1 piuttosto che da 0
      std::vector<double>& phase_results = map.at(phase);
      double avg = average_of(phase_results);
      double st_dev = standard_deviation_calc(phase_results, avg);
      if (algo == 0 && phase == 1) {
        max_y = min_y = avg;
      }
      else if (avg-st_dev < min_y) {
        min_y = avg-st_dev;
      }
      else if (avg+st_dev > max_y) {
        max_y = avg+st_dev;
      }
    }
  } 

  if (baseline_value < min_y)
    min_y = baseline_value;
  else if (baseline_value > max_y)
    max_y = baseline_value;

  std::stringstream basename;
  if (game_name == "pathfinding") {
    basename << game_name<<":"<<n_rows<<"*"<<n_columns<<"ratio("<<wall_ratio<<")horizon("<<horizon<<")random_chance("<<random_move_chance<<")"<<
    n_reps<<"rep, ("<<n_training<<") ["<<tag<<"]";
  } else {
    basename <<game_name<<":"<<n_reps<<"rep, ("<<n_training<<") ["<<tag<<"]";
  }
  std::ofstream file_dati("gnu " + basename.str() + ".txt");

  file_dati << "set terminal pngcairo size 1800,900\n";
  file_dati << "set output '"<< basename.str() << ".png'\n";
  file_dati << "set title '"<<game_name<<"'\n";
  file_dati << "set xlabel 'fase'\n";
   if (game_name == "pathfinding") {
    file_dati << "set ylabel 'efficienza relativa'\n";
  } else {
    file_dati << "set ylabel 'pr. media di vincita'\n";
  }
  file_dati << "set datafile separator ' '\n";

  file_dati << "set xrange [" << 1 << ":" << n_phases+results->size()*0.04+0.1<< "]\n";
  file_dati << "set yrange [" << min_y-0.05 << ":" << max_y+0.05 << "]\n";
  file_dati << "set xtics 1\n";
  file_dati << "set ytics 0.1\n";

  file_dati << "set key outside\n";

  file_dati << "set grid\n";
  file_dati << "set arrow from 1,"<< baseline_value <<" to "<< n_phases+results->size()*0.04+0.1 <<","<< baseline_value <<" nohead lt 2 lc 'black' dt 2\n"; //La baseline essendo una linea orizzontale possiamo mapparla con una arrow
  file_dati << "set palette model HSV defined ( 0 0 1 1, 1 1 1 1 ) \n";

  file_dati << "plot ";

  double dt;
  double pt;
  double n_eps = 0;
  double n_vbr = 0;
  double n_thompson = 0;
  double i_eps = -1;
  double i_vbr = -1;
  double i_thompson = -1;
  double palette_frac_value = 0.0;

  for (int i = 0; i < policy_vec.size(); i++) {
    if (policy_vec[i]->toString().find("Epsilon") != std::string::npos) {
      n_eps++;
    } else if (policy_vec[i]->toString().find("Thompson") != std::string::npos) {
      n_thompson++;
    }
    else  {
      n_vbr++;
    }

  }

  for (int i = 0; i < policy_vec.size(); i++) {
    if (policy_vec[i]->toString().find("Epsilon") != std::string::npos) {
      dt = 2;
      pt = 20;
      i_eps++;
      palette_frac_value = i_eps/n_eps;
    } else if (policy_vec[i]->toString().find("Thompson") != std::string::npos) {
      dt = 3;
      i_thompson++;
      palette_frac_value = i_thompson/n_thompson;
      pt = 9;
    }
    else  {
      dt = 1;
      pt = 7;
      i_vbr++;
      palette_frac_value = i_vbr/n_vbr;
    }

    file_dati << "'-' with yerrorlines title '" + policy_vec[i]->toString() + "' lt 1 lc palette frac "<< palette_frac_value <<" dt "<<dt<<" pt "<<pt<<", ";

  }

  file_dati << "1/0 t 'Baseline (random)' lt 2 lc 'black' dt 2\n"; //Creiamo una linea fittizia (1/0 non essendo calcolabile creerà una linea vuota)
            //solo per avere un nome per la baseline nella legenda (Le arrow non possono avere un nome)

  for (int algo = 0; algo < results->size(); algo++) {
    absl::flat_hash_map<int, std::vector<double>>& map = results->at(algo);
    for (int phase = 1; phase <= map.size(); phase++) { //Le fasi sono numerate da 1 piuttosto che da 0
      std::vector<double>& phase_results = map.at(phase);
      double avg = average_of(phase_results);
      double st_dev = standard_deviation_calc(phase_results, avg);
      double phase_value;
      phase_value = phase+(0.04*algo)+0.01;
      file_dati << phase_value << " " << avg << " " << st_dev << "\n";

    }
    file_dati << "e\n";
  }

  file_dati.close();

}

int main(int argc, char** argv) {
  pathfinding_parameters p_params1 = {25, 5, 5, 0.3, 0, 10};
  pathfinding_parameters p_params1R = {25, 5, 5, 0.3, 0.5, 10};
  pathfinding_parameters p_params2 = {25, 5, 5, 0.5, 0, 10};
  pathfinding_parameters p_params2R = {25, 5, 5, 0.5, 0.5, 10};

  test_parameters t_params1 = {40, 10, 30, 1, identity, "id EPS"};
  test_parameters t_params2 = {40, 10, 100, 100, identity, "id rand EPS"};
  test_parameters t_params3 = {40, 10, 100, 1, visibility_limit_no_distinction, "limit no dis EPS"};
  test_parameters t_params4 = {40, 10, 100, 1, visibility_limit_with_distinction, "limit dis EPS"};

  test_parameters t_params5 = {20, 10, 100, 1000, identity, "id"};
  test_parameters t_params6 = {20, 10, 1000, 1000, identity, "id"};

  qlearning_parameters q_params1;
  std::vector<GenericPolicy*> vecEps;
  vecEps.push_back(new EpsilonGreedyPolicy(0.01));
  vecEps.push_back(new EpsilonGreedyPolicy(0.1));
  vecEps.push_back(new EpsilonGreedyPolicy(0.2));
  vecEps.push_back(new EpsilonGreedyPolicy(0.3));
  vecEps.push_back(new EpsilonGreedyPolicy(0.4));
  vecEps.push_back(new EpsilonGreedyPolicy(0.5));
  vecEps.push_back(new EpsilonGreedyPolicy(0.6));
  vecEps.push_back(new EpsilonGreedyPolicy(0.7));
  vecEps.push_back(new EpsilonGreedyPolicy(0.8));
  vecEps.push_back(new EpsilonGreedyPolicy(0.9));
  vecEps.push_back(new EpsilonGreedyPolicy(1));

  // TestGenericGameMulti("pathfinding", vecEps, t_params1, q_params1, p_params1);
  // TestGenericGameMulti("pathfinding", vecEps, t_params2, q_params1, p_params1R);
  // TestGenericGameMulti("pathfinding", vecEps, t_params3, q_params1, p_params1);
  // TestGenericGameMulti("pathfinding", vecEps, t_params4, q_params1, p_params1);
  // TestGenericGameMulti("pathfinding", vecEps, t_params1, q_params1, p_params2);
  // TestGenericGameMulti("pathfinding", vecEps, t_params2, q_params1, p_params2R);
  // TestGenericGameMulti("pathfinding", vecEps, t_params3, q_params1, p_params2);
  // TestGenericGameMulti("pathfinding", vecEps, t_params4, q_params1, p_params2);

  // TestGenericGameMulti("blackjack", vecEps, t_params5, q_params1);
  // TestGenericGameMulti("blackjack", vecEps, t_params6, q_params1);
  // TestGenericGameMulti("tic_tac_toe", vecEps, t_params6, q_params1);

  std::vector<GenericPolicy*> vecVarioBase;
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.01, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.1, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.3, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.5, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.7, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 0.9, true));
  vecVarioBase.push_back(new VBRLikePolicyV2(2, 1, false));
  vecVarioBase.push_back(new VBRThompsonLikePolicy(2, 1, false));
  vecVarioBase.push_back(new EpsilonGreedyPolicy(0.01));

  std::vector<GenericPolicy*> vecVario8 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario8.push_back(new EpsilonGreedyPolicy(0.8));
  std::vector<GenericPolicy*> vecVario2 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario2.push_back(new EpsilonGreedyPolicy(0.2));
  std::vector<GenericPolicy*> vecVario1 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario1.push_back(new EpsilonGreedyPolicy(0.1));
  std::vector<GenericPolicy*> vecVario7 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario7.push_back(new EpsilonGreedyPolicy(0.7));
  std::vector<GenericPolicy*> vecVario4 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario4.push_back(new EpsilonGreedyPolicy(0.4));
  std::vector<GenericPolicy*> vecVario9 (vecVarioBase.begin(), vecVarioBase.end());
  vecVario9.push_back(new EpsilonGreedyPolicy(0.9));

  TestGenericGameMulti("pathfinding", vecVario1, t_params1, q_params1, p_params1);
  TestGenericGameMulti("pathfinding", vecVario7, t_params2, q_params1, p_params1R);
  TestGenericGameMulti("pathfinding", vecVario2, t_params3, q_params1, p_params1);
  TestGenericGameMulti("pathfinding", vecVario9, t_params4, q_params1, p_params1);
  TestGenericGameMulti("pathfinding", vecVario1, t_params1, q_params1, p_params2);
  TestGenericGameMulti("pathfinding", vecVario8, t_params2, q_params1, p_params2R);
  TestGenericGameMulti("pathfinding", vecVario8, t_params3, q_params1, p_params2);
  TestGenericGameMulti("pathfinding", vecVario7, t_params4, q_params1, p_params2);

  TestGenericGameMulti("blackjack", vecVario1, t_params5, q_params1);
  TestGenericGameMulti("blackjack", vecVario1, t_params6, q_params1);
  TestGenericGameMulti("tic_tac_toe", vecVario4, t_params6, q_params1);
  return 0;
}
