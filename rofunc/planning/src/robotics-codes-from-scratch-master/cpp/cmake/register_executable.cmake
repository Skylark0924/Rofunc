cmake_policy(SET CMP0072 NEW)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(EXTERNAL_LIBRARIES Eigen3::Eigen ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
set(INCLUDE_DIRECTORIES ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

function(register_executable TARGET)
  add_executable(${TARGET} "${PROJECT_SOURCE_DIR}/src/${TARGET}.cpp")
  target_link_libraries(${TARGET} ${EXTERNAL_LIBRARIES})
  target_include_directories(${TARGET} PRIVATE ${INCLUDE_DIRECTORIES})
endfunction(register_executable)