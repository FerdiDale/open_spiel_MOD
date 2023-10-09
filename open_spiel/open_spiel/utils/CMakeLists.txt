add_library (utils OBJECT
  combinatorics.h
  combinatorics.cc
  random.h
  random.cc
  serialization.h
  tensor_view.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
