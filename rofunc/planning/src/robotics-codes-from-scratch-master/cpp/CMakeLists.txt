cmake_minimum_required(VERSION 3.5)

project(robotics-codes-from-scratch)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/register_executable.cmake)

register_executable(IK)
register_executable(LQT)
register_executable(iLQR_bicopter)
register_executable(iLQR_car)
register_executable(iLQR_manipulator)
register_executable(iLQR_manipulator_obstacle)
