#!/bin/bash

path_o_spiel_src="$(pwd)/../.."

LD_LIBRARY_PATH = "$path_o_spiel_src/build"

export LD_LIBRARY_PATH

g++ -g -I "$path_o_spiel_src" -I "$path_o_spiel_src/open_spiel/abseil-cpp" -std=c++17 -o qlearn tabular_q_learning_example.cc  -L "$path_o_spiel_src/build"  -lopen_spiel -lboost_iostreams

/bin/bash
