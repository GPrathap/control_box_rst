cmake_minimum_required(VERSION 3.1)
project(control_box_rst)

set(CMAKE_CXX_STANDARD 14)

set(MESSAGE_SUPPORT OFF CACHE BOOL "Message support (downloads and compiles Protobuf)" FORCE)
set(RPC_SUPPORT OFF CACHE BOOL "Network communication (downloads and comiles gRPC, requires MESSAGE_SUPPORT" FORCE)
set(YAML_SUPPORT OFF CACHE BOOL "Yaml support" FORCE)
set(BUILD_GUI OFF CACHE BOOL "Build gui (requires QT)" FORCE)
set(BUILD_TESTS ON CACHE BOOL "Build unit tests" FORCE)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     add_compile_options(-Wno-error=maybe-uninitialized)
# endif()

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     target_compile_options(control_box_rst PRIVATE -Wno-error=maybe-uninitialized)
# endif()
set(CMAKE_CXX_FLAGS_INIT "-Werror -Wall -Wextra -Wpedantic -Wno-error=maybe-uninitialized")

# Set to Release in order to speed up the program significantly
set(CMAKE_BUILD_TYPE Release) #None, Debug, Release, RelWithDebInfo, MinSizeRel

if (BUILD_TESTING)
  message(STATUS "Tests are currently deactivated.")
  #set(BUILD_TESTS ON)
endif()

if(CATKIN_ENABLE_TESTING AND NOT CATKIN_SKIP_TESTING) 
  message(STATUS "Catkin testing is currently deactivated.")
  #set(BUILD_TESTS ON)
endif() 

add_subdirectory(src)

# Install catkin package.xml
install(FILES package.xml DESTINATION share/control_box_rst)
