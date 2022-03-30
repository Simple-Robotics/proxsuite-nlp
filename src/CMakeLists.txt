
function(create_library)
  # get the compile defs
  get_directory_property(CURRENT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)

  add_library(${PROJECT_NAME} SHARED ${HEADERS})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${CURRENT_COMPILE_DEFINITIONS})
  
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
  if(BUILD_LIBRARY_WITH_VERSION_SUFFIX)
    TAG_LIBRARY_VERSION(${PROJECT_NAME})
  endif()

  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    add_definitions(-DWITH_PINOCCHIO_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)

  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::filesystem Boost::system Boost::serialization)
  target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
  target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
  set(INSTALL_DIRS_FROM_INTERFACE $<INSTALL_INTERFACE:include>)

endfunction()

create_library()

ADD_HEADER_GROUP(HEADERS)

