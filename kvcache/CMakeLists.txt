cmake_minimum_required(VERSION 3.14)
project(paged_kv_cache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(kvcache src/kv_cache_manager.cpp)
target_include_directories(kvcache PUBLIC include)

add_executable(kv_sim src/simulate.cpp)
target_link_libraries(kv_sim kvcache)

add_executable(kv_tests tests/test_kv_cache.cpp)
target_link_libraries(kv_tests kvcache)

enable_testing()
add_test(NAME kv_tests COMMAND kv_tests)
