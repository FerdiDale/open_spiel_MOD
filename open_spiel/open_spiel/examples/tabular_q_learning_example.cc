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

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_q_learning.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "MYpolicies/generic_policy.h"
#include "MYpolicies/eps_greedy.h"
#include "MYpolicies/VBR_like_v1.h"
#include "MYpolicies/VBR_like_v2.h"

#include <iostream>
#include "gnuplot-iostream.h"

using policies::GenericPolicy;
using policies::EpsilonGreedyPolicy;
using policies::VBRLikePolicyV1;
using policies::VBRLikePolicyV2;

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::TurnBasedSimultaneousGame;

using open_spiel::algorithms::TabularQLearningSolver;

void SampleUntilNextStateOrTerminal(State* state, std::mt19937* rng_) {
  while (state->IsChanceNode() && !state->IsTerminal()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    state->ApplyAction(open_spiel::SampleAction(outcomes, *rng_).first);
  }
}

Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State>& state) {

  std::vector<Action> legal_actions = state->LegalActions();
  Action optimal_action = open_spiel::kInvalidAction;

  double value = -1;
  for (const Action& action : legal_actions) {
    double q_val = q_values[{state->ToString(), action}];
    if (q_val >= value) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

void SolveTicTacToe() {

  std::mt19937 rng_;
  //
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  TabularQLearningSolver tabular_q_learning_solver(game);

  for (int phase = 0; phase < 10; phase++) { //Ripetiamo il test in 10 fasi per notare l'evoluzione dei risultati al miglioramento della tabella

    std::cout<<"FASE NUMERO "<<phase+1<<std::endl;

    for (int iter = 0; iter < 10000; iter++) { //Eseguiamo 10mila iterazioni in cui addestriamo l'agente
      tabular_q_learning_solver.RunIteration();
    }
    //
    // n_wins = 0;
    absl::flat_hash_map<std::pair<std::string, Action>, double> QTable = tabular_q_learning_solver.GetQValueTable();
    //
    // for (int match = 0; match < 1000; match++) { //L'agente gioca al suo meglio 1000 volte, per avere una stima accurata della sua bravura
    //   std::unique_ptr<State> state = game->NewInitialState();
    //   while (!state->IsTerminal()) {
    //     // Action optimal_action = GetOptimalAction(QTable, state);
    //     // state->ApplyAction(optimal_action);
    //     if (state->CurrentPlayer() != 0) {
    //       std::vector<Action> legal_actions = state->LegalActions();
    //       Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
    //       state->ApplyAction(random_action);
    //     }
    //     else {
    //       Action optimal_action = GetOptimalAction(QTable, state);
    //       state->ApplyAction(optimal_action);
    //     }
    //   }
    //   std::cout<<"RITORNO "<<state->Returns()[0]<<std::endl;
    //   if (state->Returns()[0] == game->MaxUtility())
    //     n_wins++;
    // }
    //
    // win_percentage = n_wins/1000;
    // phase_scores.push_back(std::make_tuple(algo_id, phase+1, win_percentage));

  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout<<state->ToString()<<std::endl;
    std::cout<<state->CurrentPlayer()<<std::endl;

    if (state->CurrentPlayer() != 0) {
      std::vector<Action> legal_actions = state->LegalActions();
      Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
      state->ApplyAction(random_action);
    }
    else {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
    }
  }

  std::cout<<"PUNTEGGIO: GIOCATORE 0 PUNTI "<<state->Returns()[0]<<" GIOCATORE 1 PUNTI "<<state->Returns()[1]<<std::endl;
}

void SolveChess() {

  std::mt19937 rng_;

  std::shared_ptr<const Game> game = open_spiel::LoadGame("chess");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 10;
  while (iter-- > 0) {
    std::cout<<iter<<std::endl;
    tabular_q_learning_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    if (state->CurrentPlayer() == 0) {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
    }
    else {
      std::vector<Action> legal_actions = state->LegalActions();
      Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
      state->ApplyAction(random_action);
    }
  }

  // Tie.
  SPIEL_CHECK_EQ(state->Rewards()[1], 1);
  SPIEL_CHECK_EQ(state->Rewards()[0], -1);
}


void SolveBackgammon() {

  std::mt19937 rng_;

  std::shared_ptr<const Game> game = open_spiel::LoadGame("backgammon");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 40;
  while (iter-- > 0) {
    std::cout<<iter<<std::endl;
    tabular_q_learning_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout<<state->ToString()<<std::endl;
    std::cout<<state->CurrentPlayer()<<std::endl;

    if (state->CurrentPlayer() != 0) {
      std::vector<Action> legal_actions = state->LegalActions();
      Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
      state->ApplyAction(random_action);
    }
    else {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
    }
  }

  // Tie.
  std::cout<<"PUNTEGGIO: GIOCATORE 0 PUNTI "<<state->Returns()[0]<<" GIOCATORE 1 PUNTI "<<state->Returns()[1]<<std::endl;
}

void SolvePoker() {

  static std::random_device rd;
  static std::mt19937 rng_ (rd());

  std::shared_ptr<const Game> game = open_spiel::LoadGame("leduc_poker");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 1000;
  while (iter-- > 0) {
    std::cout<<iter<<std::endl;
    tabular_q_learning_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double> q_values =
      tabular_q_learning_solver.GetQValueTable();


  int wins = 0;
  int draws = 0;
  int losses = 0;

  for (int i = 0; i<30; i++){
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      // std::cout<<state->ToString()<<std::endl;
      // std::cout<<state->CurrentPlayer()<<std::endl;

      if (state->CurrentPlayer() != 0) {
        std::vector<Action> legal_actions = state->LegalActions();
        Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
        state->ApplyAction(random_action);
      }
      else {
        Action optimal_action = GetOptimalAction(q_values, state);
        state->ApplyAction(optimal_action);
      }
    }

    if (state->Returns()[0] > state->Returns()[1])
      wins++;
    else if (state->Returns()[0] == state->Returns()[1])
      draws++;
    else
      losses++;

  }

  std::cout<<"VITTORIE: "<<wins<<", SCONFITTE: "<<losses<<", PAREGGI: "<<draws<<std::endl;

}

void SolveGoofspiel(){
  static std::random_device rd;
  static std::mt19937 rng_ (rd());

  std::shared_ptr<const Game> game = open_spiel::LoadGameAsTurnBased("goofspiel");

  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 100;
  while (iter-- > 0) {
    std::cout<<iter<<std::endl;
    tabular_q_learning_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double> q_values =
      tabular_q_learning_solver.GetQValueTable();


  int wins = 0;
  int draws = 0;
  int losses = 0;

  for (int i = 0; i<30; i++){
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      // std::cout<<state->ToString()<<std::endl;
      // std::cout<<state->CurrentPlayer()<<std::endl;

      if (state->CurrentPlayer() != 0) {
        std::vector<Action> legal_actions = state->LegalActions();
        Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
        state->ApplyAction(random_action);
      }
      else {
        Action optimal_action = GetOptimalAction(q_values, state);
        state->ApplyAction(optimal_action);
      }
    }

    if (state->Returns()[0] > state->Returns()[1])
      wins++;
    else if (state->Returns()[0] == state->Returns()[1])
      draws++;
    else
      losses++;

  }

  std::cout<<"VITTORIE: "<<wins<<", SCONFITTE: "<<losses<<", PAREGGI: "<<draws<<std::endl;

}

void SolvePig(){
  static std::random_device rd;
  static std::mt19937 rng_ (rd());

  std::shared_ptr<const Game> game = open_spiel::LoadGameAsTurnBased("pig");

  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 100;
  while (iter-- > 0) {
    std::cout<<iter<<std::endl;
    tabular_q_learning_solver.RunIteration();
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double> q_values =
      tabular_q_learning_solver.GetQValueTable();


  int wins = 0;
  int draws = 0;
  int losses = 0;

  for (int i = 0; i<30; i++){
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      // std::cout<<state->ToString()<<std::endl;
      // std::cout<<state->CurrentPlayer()<<std::endl;

      if (state->CurrentPlayer() != 0) {
        std::vector<Action> legal_actions = state->LegalActions();
        Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
        state->ApplyAction(random_action);
      }
      else {
        Action optimal_action = GetOptimalAction(q_values, state);
        state->ApplyAction(optimal_action);
      }
    }

    if (state->Returns()[0] > state->Returns()[1])
      wins++;
    else if (state->Returns()[0] == state->Returns()[1])
      draws++;
    else
      losses++;

  }

  std::cout<<"VITTORIE: "<<wins<<", SCONFITTE: "<<losses<<", PAREGGI: "<<draws<<std::endl;

}

void Solve2048() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("2048");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 1;
  while (iter-- > 0) {
    tabular_q_learning_solver.RunIteration();
    std::cout<<"iterazione "<<iter<<std::endl;
  }

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    Action optimal_action = GetOptimalAction(q_values, state);
    state->ApplyAction(optimal_action);
    std::cout<<"AZIONE"<<std::endl;
  }

      std::cout<<"FINE"<<std::endl;


  SPIEL_CHECK_GE(state->Returns()[0], 2048);
}

void SolveTicTacToeEligibilityTraces() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::TabularQLearningSolver
      tabular_q_learning_solver_lambda00(game, -1.0, 0.0001, 0.01, 0.99, 0.0);
  open_spiel::algorithms::TabularQLearningSolver
      tabular_q_learning_solver_lambda01(game, -1.0, 0.0001, 0.001, 0.99, 0.1);

  int count_tie_games_lambda00 = 0;
  int count_tie_games_lambda01 = 0;
  for (int i = 1; i < 10000; i++) {
    tabular_q_learning_solver_lambda00.RunIteration();

    const absl::flat_hash_map<std::pair<std::string, Action>, double>&
        q_values_lambda00 = tabular_q_learning_solver_lambda00.GetQValueTable();
    std::unique_ptr<State> state = game->NewInitialState();

    while (!state->IsTerminal()) {
      state->ApplyAction(GetOptimalAction(q_values_lambda00, state));
    }

    count_tie_games_lambda00 += state->Rewards()[0] == 0 ? 1 : 0;
  }

  for (int i = 1; i < 10000; i++) {
    tabular_q_learning_solver_lambda01.RunIteration();

    const absl::flat_hash_map<std::pair<std::string, Action>, double>&
        q_values_lambda01 = tabular_q_learning_solver_lambda01.GetQValueTable();
    std::unique_ptr<State> state = game->NewInitialState();

    while (!state->IsTerminal()) {
      state->ApplyAction(GetOptimalAction(q_values_lambda01, state));
    }

    count_tie_games_lambda01 += state->Rewards()[0] == 0 ? 1 : 0;
  }

  //  Q-Learning(0.1) gets equilibrium faster than Q-Learning(0.0).
  //  More ties in the same amount of time.
  SPIEL_CHECK_GT(count_tie_games_lambda01, count_tie_games_lambda00);
}

void SolveCatch() {

  std::shared_ptr<const Game> game = open_spiel::LoadGame("catch");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int training_iter = 100000;
  while (training_iter-- > 0) {
    tabular_q_learning_solver.RunIteration();
  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();

  int eval_iter = 1000;
  int total_reward = 0;
  while (eval_iter-- > 0) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
      total_reward += state->Rewards()[0];
    }
  }

  SPIEL_CHECK_GT(total_reward, 0);
}

void TestGenericSinglePlayerGame(std::string chosen_game_name) {
  std::shared_ptr<const Game> game = open_spiel::LoadGameAsTurnBased(chosen_game_name);
  TabularQLearningSolver QLearningVBRLike2v1(game, new VBRLikePolicyV2(2, true));
  TabularQLearningSolver QLearningVBRLike2v2(game, new VBRLikePolicyV2(2, false));
  TabularQLearningSolver QLearningEpsilonGreedy(game);

  std::random_device rd;
  std::mt19937 rng_(rd());

  std::vector<TabularQLearningSolver> vec_algos;
  vec_algos.push_back(QLearningVBRLike2v1);
  vec_algos.push_back(QLearningVBRLike2v2);
  vec_algos.push_back(QLearningEpsilonGreedy);

  double n_wins;
  double win_percentage;
  std::vector<std::tuple<int, int, double>> phase_scores;

  for (int algo_id = 0; algo_id < vec_algos.size();  algo_id++) {
    for (int phase = 0; phase < 10; phase++) { //Ripetiamo il test in 10 fasi per notare l'evoluzione dei risultati al miglioramento della tabella

      std::cout<<"FASE NUMERO "<<phase+1<<std::endl;

      for (int iter = 0; iter < 1000; iter++) { //Eseguiamo 10mila iterazioni in cui addestriamo l'agente
        std::cout<<iter<<std::endl;
        vec_algos[algo_id].RunIteration();
      }

      n_wins = 0;
      absl::flat_hash_map<std::pair<std::string, Action>, double> QTable = vec_algos[algo_id].GetQValueTable();
      // std::cout<<"POST TRASFERIMENTO"<<std::endl;
      // for (const auto&[k,v] : QTable) {
      //   std::cout<<"STATO "<<k.first<<" AZIONE "<<k.second<<" QVALUE "<<v<<std::endl;
      // }

      for (int match = 0; match < 1000; match++) { //L'agente gioca al suo meglio 1000 volte, per avere una stima accurata della sua bravura
        std::unique_ptr<State> state = game->NewInitialState();
        while (!state->IsTerminal()) {
          SampleUntilNextStateOrTerminal(state.get(), &rng_);
          std::cout<<state->ToString()<<std::endl;
          Action optimal_action = GetOptimalAction(QTable, state);
          std::cout<<"AZIONE "<<optimal_action<<std::endl;
          state->ApplyAction(optimal_action);
          SampleUntilNextStateOrTerminal(state.get(), &rng_);
          // if (state->CurrentPlayer() != 0) {
          //   // std::cout<<"STATO "<<state->ToString()<<std::endl;
          //   // for (Action a : state->LegalActions()) {
          //   //   std::cout<<"AZIONE "<<a<<" QVALUE "<<QTable[{state->ToString(), a}]<<std::endl;
          //   // }
          //   std::vector<Action> legal_actions = state->LegalActions();
          //   Action random_action = legal_actions[absl::Uniform<int>(rng_, 0, legal_actions.size())];
          //   state->ApplyAction(random_action);
          // }
          // else {
          //   // std::cout<<"STATO "<<state->ToString()<<std::endl;
          //   // for (Action a : state->LegalActions()) {
          //   //   std::cout<<"AZIONE "<<a<<" QVALUE "<<QTable[{state->ToString(), a}]<<std::endl;
          //   // }
          //   Action optimal_action = GetOptimalAction(QTable, state);
          //   state->ApplyAction(optimal_action);
          // }
        }
        std::cout<<"RITORNO "<<state->Returns()[0]<<std::endl;
        // if (state->Returns()[0] == game->MaxUtility())
        //   n_wins++;
        if (state->Returns()[0] >= 0)
          n_wins++;
      }

      win_percentage = n_wins/1000;
      phase_scores.push_back(std::make_tuple(algo_id, phase+1, win_percentage));

    }

  }

  for (auto tupla : phase_scores) {
    std::cout<<"UNO "<<std::get<0>(tupla)<<" DUE "<<std::get<1>(tupla)<<" TRE "<<std::get<2>(tupla)<<std::endl;
  }

  Gnuplot gp;
  gp << "set terminal png size 1600,800\n";
  gp << "set output 'data_plot.png'\n";
  gp << "set title 'Data Plot'\n";
  gp << "set xlabel 'fase'\n";
  gp << "set ylabel '% di vittorie'\n";
  gp << "set datafile separator ' '\n";

  gp << "set xrange [" << 1 << ":" << 10 << "]\n";
  gp << "set yrange [" << 0 << ":" << 1 << "]\n";
  gp << "set ytics 0.1\n";
  gp << "set grid\n";

  gp << "plot '-' with linespoints title 'VBRLike2 (history)' lt 1 lc 'red' dt 2 pt 7, \
          '-' with linespoints title 'VBRLike2 (no history)' lt 2 lc 'blue' dt 2 pt 7, \
          '-' with linespoints title 'EpsilonGreedy' lt 2 lc 'green' dt 2 pt 7\n";
  for (size_t i = 0; i < phase_scores.size(); ++i) {
    if (std::get<0>(phase_scores[i]) == 0)
      gp << std::get<1>(phase_scores[i]) << " " << std::get<2>(phase_scores[i]) << "\n";
  }
  gp << "e\n";

  for (size_t i = 0; i < phase_scores.size(); ++i) {
    if (std::get<0>(phase_scores[i]) == 1)
      gp << std::get<1>(phase_scores[i]) << " " << std::get<2>(phase_scores[i]) << "\n";
  }
  gp << "e\n";

  for (size_t i = 0; i < phase_scores.size(); ++i) {
    if (std::get<0>(phase_scores[i]) == 2)
      gp << std::get<1>(phase_scores[i]) << " " << std::get<2>(phase_scores[i]) << "\n";
  }
  gp << "e\n";

}

int main(int argc, char** argv) {

  // SolveBackgammon();
  // SolveTicTacToe();
  // SolvePoker();
  // SolveGoofspiel();
  // SolvePig();
  // SolveChess();
  //SolveTicTacToeEligibilityTraces();
  //SolveCatch();
  //Solve2048();
  TestGenericSinglePlayerGame("pathfinding");
  //CLIFF WALKING
  //PATHFINDING
  //DEEP SEA
  //SOLITAIRE + BLACKJACK

  return 0;
}
