#!/bin/bash

path_o_spiel_src="$(pwd)/../.."

g++ -g -o3 -I "$path_o_spiel_src" -I "$path_o_spiel_src/open_spiel/abseil-cpp" -std=c++17 -o qlearn tabular_q_learning_example.cc  -L "$path_o_spiel_src/build"  -lopen_spiel

