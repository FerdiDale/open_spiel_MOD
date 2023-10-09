add_library (algorithms OBJECT
  tabular_q_learning.cc
  tabular_q_learning.h
)
target_include_directories (algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
