cmake_minimum_required(VERSION 3.8)
project(sercom)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_path(asio_INCLUDE_DIRS "asio.hpp")
if(NOT asio_INCLUDE_DIRS)
  message(FATAL_ERROR "ASIO not found. Please install libasio-dev")
endif()

add_executable(run_sercom src/run_sercom.cpp)

target_include_directories(run_sercom PRIVATE
  ${asio_INCLUDE_DIRS}
)

ament_target_dependencies(run_sercom
  rclcpp
  geometry_msgs
  std_msgs
)

install(TARGETS run_sercom
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rclcpp geometry_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
