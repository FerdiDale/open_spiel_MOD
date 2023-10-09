#ifndef STATE_ABS_FUNCS_H
#define STATE_ABS_FUNCS_H

#include <algorithm>
#include <random>

#include <iostream>
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"

namespace policies {
  
std::string visibility_limit_no_distinction(const std::string state_str);

std::string visibility_limit_with_distinction(const std::string state_str);

std::string identity (const std::string str);

}

#endif
