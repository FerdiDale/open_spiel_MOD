add_executable(tabular_q_learning_example tabular_q_learning_example.cc ${OPEN_SPIEL_OBJECTS})

if (OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  target_link_libraries(alpha_zero_example TensorflowCC::TensorflowCC)
endif ()

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  target_link_libraries (alpha_zero_torch_example ${TORCH_LIBRARIES})

  target_link_libraries (alpha_zero_torch_game_example ${TORCH_LIBRARIES})

  target_link_libraries (dqn_torch_example ${TORCH_LIBRARIES})
endif ()

if (BUILD_SHARED_LIB)
  target_link_libraries(shared_library_example open_spiel)
endif()
