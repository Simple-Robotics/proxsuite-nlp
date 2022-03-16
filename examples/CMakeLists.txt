# Copyright (C) 2022 LAAS-CNRS / INRIA

file(GLOB CPP_EXAMPLES
     "*.cpp")

message(STATUS "CPP EXAMPLES: ${CPP_EXAMPLES}")


function(add_lienlp_example exfile)
  get_filename_component(example_name ${exfile} NAME_WE)
  set(example_name "example-${example_name}")
  get_filename_component(exfile ${exfile} NAME)
  message(STATUS "Adding executable ${example_name} (file ${exfile})")

  ADD_UNIT_TEST(${example_name} ${exfile})
  set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${example_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${example_name} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${example_name} PRIVATE fmt::fmt)
endfunction()

foreach(exfile ${CPP_EXAMPLES})
  add_lienlp_example(${exfile})
endforeach()

