set(GAME_SOURCES
  blackjack.cc
  blackjack.h
  pathfinding.cc
  pathfinding.h
  tic_tac_toe.cc
  tic_tac_toe.h
)

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  set(GAME_SOURCES ${GAME_SOURCES} hanabi.cc hanabi.h)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  set(GAME_SOURCES ${GAME_SOURCES} universal_poker.cc universal_poker.h)
endif()

add_library (games OBJECT ${GAME_SOURCES})

target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  add_subdirectory(hanabi)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_subdirectory(universal_poker)
endif()
if (${OPEN_SPIEL_BUILD_WITH_GAMUT})
  add_subdirectory(gamut)
endif()

