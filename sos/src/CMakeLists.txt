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

add_library(sos sos_fn.c sos.c sos_schema.c sos_config.c sos_index.c sos_key.c sos_iter.c sos_value.c)
target_link_libraries(sos ods dl pthread)

add_executable(sos_cmd sos_cmd.c)
target_link_libraries(sos_cmd sos yaml)

INSTALL(TARGETS sos sos_cmd
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib)
