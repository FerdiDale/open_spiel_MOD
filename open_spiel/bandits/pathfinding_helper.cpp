
#include <algorithm>
#include <random>

#include "pathfinding_helper.h"

namespace policies {

std::vector<std::pair<int, int>> possible_next_positions(std::vector<std::vector<char>> maze, std::pair<int, int> curr) {
  std::vector<std::pair<int, int>> ret;
  int x = curr.first;
  int y = curr.second;

  if (x != 0 && maze[x-1][y] != '*')
    ret.push_back({x-1, y});
  if (y != 0 && maze[x][y-1] != '*')
    ret.push_back({x, y-1});
  if (x != maze.size()-1 && maze[x+1][y] != '*')
    ret.push_back({x+1, y});
  if (y != maze[0].size()-1 && maze[x][y+1] != '*')
    ret.push_back({x, y+1});
  return ret;
}

bool DFS (std::vector<std::vector<char>> maze, std::vector<std::vector<char>>* colors_p, std::pair<int, int> curr) {
  (*colors_p)[curr.first][curr.second] = 'n';
  if (maze[curr.first][curr.second] == 'A') //Target del labirinto
    return true;
  else {
    bool ret = false;
    for (auto& [next_x, next_y] : possible_next_positions(maze, curr)) {
      if ((*colors_p)[next_x][next_y] == 'b')
        ret = ret || DFS(maze, colors_p, {next_x, next_y});
    }
    return ret;
  }
}

std::string maze_to_string(std::vector<std::vector<char>> maze) {
  std::string ret = "";
  for (int i = 0; i < maze.size(); i++) {
    for (int j = 0; j < maze[i].size(); j++) {
      ret+=maze[i][j];
    }
    ret+='\n';
  }
  return ret;
}

bool traversable_maze_check(std::vector<std::vector<char>> maze, std::pair<int, int> source) {
  std::vector<std::vector<char>> colors;
  colors.resize(maze.size());
  for (int i = 0; i < maze.size(); i++) {
    colors[i].resize(maze[i].size());
    for (int j = 0; j < maze[i].size(); j++) {
      colors[i][j] = 'b';
    }
  }
  bool ret = DFS (maze, &colors, source);
  return ret;
}

std::string maze_gen (int n_rows , int n_columns, double wall_ratio) {

  std::random_device rd;
  std::mt19937 rng_(rd());

  std::vector<std::vector<char>> maze;
  int source_x;
  int source_y;

  maze.resize(n_rows);

  do {

    for (int i = 0; i < n_rows; i++) { //Riempiamo di spazi vuoti il labirinto
      maze[i].resize(n_columns);
      for (int j = 0; j < n_columns; j++) {
        maze[i][j] = '.';
      }
    }

    source_x = 0; //Scegliamo la posizione iniziale
    source_y = 0;
    maze[source_x][source_y] = 'a';

    int target_x = n_rows-1; //Scegliamo la posizione dell'obiettivo
    int target_y = n_columns-1;
    maze[target_x][target_y] = 'A';


    for (int i = 0; i < n_rows*n_columns*wall_ratio; i++) {
      int wall_x;
      int wall_y;

      do {
        wall_x = absl::Uniform<int>(rng_, 0, n_rows);
        wall_y = absl::Uniform<int>(rng_, 0, n_columns);
      } while (maze[wall_x][wall_y] != '.'); //Il muro è in una posizione giusta solo se sostituisce uno spazio vuoto DA CAMBIARE

      maze[wall_x][wall_y] = '*';
    }

  } while (!traversable_maze_check(maze, {source_x, source_y})); //Passiamo l'origine della dfs per scoprire se il target è raggiungibile


  return (maze_to_string(maze));

}

std::vector<std::vector<char>> parseStringGrid(std::string grid_str, std::pair<int, int>& start_point, std::pair<int, int>& end_point) {
  std::vector<std::vector<char>> grid_vec;
  int row = 0;
  int col = 0;
  std::vector<char> new_row;
  grid_vec.push_back(new_row);

  for (auto c : grid_str) {
    if (c == 'a') {
      start_point = {row, col};
    }
    if (c == 'A') {
      end_point = {row, col};
    }
    if (c == '\n') {
      row += 1;
      col = 0;
      grid_vec.push_back(new_row);
    } else {
      grid_vec[row].push_back(c);
      col += 1;
    }
  }

  if (grid_str.back() == '\n')
    grid_vec.pop_back();

  return grid_vec;
}

int BFS (std::string maze) {
  std::pair<int, int> starting_position;
  std::pair<int, int> destination;
  std::vector<std::vector<char>> grid_vec = parseStringGrid(maze, starting_position, destination);
  int n_rows = grid_vec.size();
  int n_columns = grid_vec[0].size();
  char color [n_rows][n_columns];
  int dist [n_rows][n_columns];
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_columns; j++) {
      color[i][j] = 'b';
      dist[i][j] = -1;
    }
  }
  std::queue<std::pair<int, int>> pos_queue;
  std::pair<int, int> curr_pos;
  pos_queue.push(starting_position);
  color[starting_position.first][starting_position.second] = 'g';
  dist[starting_position.first][starting_position.second] = 0;

  while (!pos_queue.empty()) {

    curr_pos = pos_queue.front();
    pos_queue.pop();
    for (auto& [x,y] : possible_next_positions(grid_vec, curr_pos)) {
      if (color[x][y] == 'b') {
        pos_queue.push({x,y});
        color[x][y] = 'g';
        dist[x][y] = dist[curr_pos.first][curr_pos.second]+1;
      }
    }

    color[curr_pos.first][curr_pos.second] = 'n';

  }

  return dist[destination.first][destination.second];
  
}

}