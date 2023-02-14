add_executable("duneuro_meg_forward_test" duneuro_meg_forward_test.cc)
target_link_dune_default_libraries("duneuro_meg_forward_test")

# create symlinks to volume conductor and dipole files
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/configs.ini" "${CMAKE_CURRENT_BINARY_DIR}/configs.ini")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/mesh.msh" "${CMAKE_CURRENT_BINARY_DIR}/mesh.msh")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/conductivities.txt" "${CMAKE_CURRENT_BINARY_DIR}/conductivities.txt")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/dipole.txt" "${CMAKE_CURRENT_BINARY_DIR}/dipole.txt")
