cmake_minimum_required(VERSION 2.6)

include_directories(
	"${PROJECT_BINARY_DIR}/../../"
	"${PROJECT_SOURCE_DIR}/include"
	"${PROJECT_SOURCE_DIR}/../ods/include"
)

add_library(sos sos_fn.c sos.c sos_key.c sos_iter.c)
target_link_libraries(sos ods dl pthread)

# add_executable(sos_test sos_test.c)
# target_link_libraries(sos_test sos yaml)

add_executable(sos_cmd sos_cmd.c)
target_link_libraries(sos_cmd sos yaml)

