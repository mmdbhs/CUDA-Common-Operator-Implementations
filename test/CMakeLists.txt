find_package(GTest REQUIRED)
find_package(glog REQUIRED)

project(test_layer)

set(link_lib glog::glog GTest::gtest)

aux_source_directory(../test DIR_tests)

message("DIR_tests" ${DIR_tests})
add_executable(test_layer ${DIR_tests})

target_link_libraries(test_layer ${link_lib} gtest)
target_link_directories(test_layer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_layer cuda_layer)


target_include_directories(test_layer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_layer PUBLIC ${GTest_INCLUDE_DIR})
