#!/bin/bash

g++ -g -I /home/ferdi/Desktop/Uni/OpenSpiel/open_spiel -I /home/ferdi/Desktop/Uni/OpenSpiel/open_spiel/open_spiel/abseil-cpp         -std=c++17 -o qlearn tabular_q_learning_example.cc         -L /home/ferdi/Desktop/Uni/OpenSpiel/open_spiel/build  -lopen_spiel -lboost_iostreams
