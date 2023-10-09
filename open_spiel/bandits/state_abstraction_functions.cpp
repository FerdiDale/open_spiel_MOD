#include "state_abstraction_functions.h"

namespace policies {
  
std::string visibility_limit_no_distinction(const std::string state_str) {

  std::cout<<"ORIGINALE "<<std::endl<<state_str<<std::endl<<std::endl;

  absl::flat_hash_map<std::pair<int, int>, char> grid_layout;

  int curr_column = 0;
  int curr_row = 0;
  std::pair<int,int> player_pos;

  for (auto& c : state_str) {
    grid_layout[{curr_row, curr_column}] = c;
    if (c == '\n') {
      curr_row++;
      curr_column = 0;
    }
    else {
      if (c == '0')
        player_pos = {curr_row, curr_column};
      curr_column++;
    }
  }

  absl::flat_hash_map<std::pair<int, int>, char>::iterator it;

  it = grid_layout.find({player_pos.first-1, player_pos.second});
  char charN = (it == grid_layout.end() ? '*' : it->second);

  it = grid_layout.find({player_pos.first, player_pos.second-1});
  char charW = (it == grid_layout.end() ? '*' : it->second);

  it = grid_layout.find({player_pos.first, player_pos.second+1});
  char charE = (it == grid_layout.end() ? '*' : it->second);

  it = grid_layout.find({player_pos.first+1, player_pos.second});
  char charS = (it == grid_layout.end() ? '*' : it->second);

  std::string ret;
  ret+=" ";
  if (charN == '*' || charN == '.')
    ret+=charN;
  else
    ret+='*';
  ret+= "\n";
  if (charW == '*' || charW == '.')
    ret+=charW;
  else
    ret+="*";
  ret += "0";
  if (charE == '*' || charE == '.')
    ret+=charE;
  else
    ret+='*';
  ret+="\n ";
  if (charS == '*' || charS == '.')
    ret+=charS;
  else
    ret+='*';

  std::cout<<"ASTRAZIONE "<<std::endl<<ret<<std::endl<<std::endl;

  return ret;

}

std::string visibility_limit_with_distinction(const std::string state_str) {

  absl::flat_hash_map<std::pair<int, int>, char> grid_layout;

  int curr_column = 0;
  int curr_row = 0;
  std::pair<int,int> player_pos;

  for (auto& c : state_str) {
    grid_layout[{curr_row, curr_column}] = c;
    if (c == '\n') {
      curr_row++;
      curr_column = 0;
    }
    else {
      if (c == '0')
        player_pos = {curr_row, curr_column};
      curr_column++;
    }
  }

  absl::flat_hash_map<std::pair<int, int>, char>::iterator it;

  it = grid_layout.find({player_pos.first-1, player_pos.second});
  char charN = (it == grid_layout.end() ? '\n' : it->second);

  it = grid_layout.find({player_pos.first, player_pos.second-1});
  char charW = (it == grid_layout.end() ? '\n' : it->second);

  it = grid_layout.find({player_pos.first, player_pos.second+1});
  char charE = (it == grid_layout.end() ? '\n' : it->second);

  it = grid_layout.find({player_pos.first+1, player_pos.second});
  char charS = (it == grid_layout.end() ? '\n' : it->second);

  std::string ret;
  ret+=" ";
  if (charN == '*' || charN == '.')
    ret+=charN;
  ret+= "\n";
  if (charW == '*' || charW == '.')
    ret+=charW;
  ret += "0";
  if (charE == '*' || charE == '.')
    ret+=charE;
  ret+="\n ";
  if (charS == '*' || charS == '.')
    ret+=charS;

  return ret;

}

std::string identity (const std::string str) {
  return str;
}


}
