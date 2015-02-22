cmake_minimum_required(VERSION 2.6)

include_directories(
  "${PROJECT_BINARY_DIR}/../../"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/../ods/include"
  "${ZAP_PATH}/include"
  )

link_directories(
  "${ZAP_PATH}/lib"
)

add_library(sos sos_fn.c sos.c sos_key.c sos_iter.c sos_value.c)
target_link_libraries(sos ods dl pthread)

# add_executable(sos_test sos_test.c)
# target_link_libraries(sos_test sos yaml)

# add_executable(sosd sosd.c)
# target_link_libraries(sosd sos zap)

add_executable(sos_cmd sos_cmd.c)
target_link_libraries(sos_cmd sos yaml)

add_executable(job_add job_add.c)
target_link_libraries(job_add sos)

add_executable(job_query job_query.c)
target_link_libraries(job_query sos)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/sos_cmd DESTINATION bin)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/libsos.so DESTINATION lib)


