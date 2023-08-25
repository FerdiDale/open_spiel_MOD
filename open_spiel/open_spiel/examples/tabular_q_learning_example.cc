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

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::TurnBasedSimultaneousGame;

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

  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

//SERVE PER TOTALLY RANDOM 30000 ITER
//PER EPSILON GREEDY BASTANO 4?
//PER VBRV1 50000
//VBRV2 40000?
  int iter = 10;
  while (iter-- > 0) {
    tabular_q_learning_solver.RunIteration();
    std::cout<<iter<<std::endl;
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

int main(int argc, char** argv) {

  // SolveBackgammon();
  // SolveTicTacToe();
  // SolvePoker();
  SolveGoofspiel();
  // SolveChess();
  //SolveTicTacToeEligibilityTraces();
  //SolveCatch();
  //Solve2048();

  return 0;
}
