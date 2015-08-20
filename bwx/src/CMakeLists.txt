cmake_minimum_required(VERSION 2.6)

include_directories(
  "${PROJECT_BINARY_DIR}/../../"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/../sos/include"
  "${PROJECT_SOURCE_DIR}/../ods/include"
  )

add_executable(bwx_job_add bwx_job_add.c)
target_link_libraries(bwx_job_add sos)

add_executable(bwx_import bwx_import.c)
target_link_libraries(bwx_import sos)

add_executable(bwx_job_ls bwx_job_ls.c)
target_link_libraries(bwx_job_ls sos m)

INSTALL(TARGETS bwx_job_add bwx_job_ls bwx_import
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib)
