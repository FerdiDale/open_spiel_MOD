add_library (game_transforms OBJECT
  turn_based_simultaneous_game.cc
  turn_based_simultaneous_game.h
)
target_include_directories (game_transforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
