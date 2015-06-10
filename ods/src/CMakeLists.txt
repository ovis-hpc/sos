cmake_minimum_required(VERSION 2.6)

include_directories(
	"${PROJECT_BINARY_DIR}/../../"
	"${PROJECT_SOURCE_DIR}/include"
)

add_library(ods ods_idx.c ods.c rbt.c)
add_library(idx_BPTREE bpt.c)
add_library(idx_BXTREE bxt.c)
add_library(key_BLKMAP key_blkmap.c)
add_library(key_SHA256 key_sha256.c)
add_library(key_STRING key_str.c)
add_library(key_UINT96 key_uint96.c)
add_library(key_UINT64 key_uint64.c)
add_library(key_UINT32 key_uint32.c)
add_library(key_DOUBLE key_double.c)

target_link_libraries(ods dl pthread)
target_link_libraries(idx_BPTREE ods)
target_link_libraries(idx_BXTREE ods)

add_executable(idx_test idx_test.c)
target_link_libraries(idx_test ods)
add_executable(rand_test rand_test.c)
target_link_libraries(rand_test ods)

INSTALL(TARGETS ods
		idx_BPTREE
		idx_BXTREE
		key_BLKMAP
		key_SHA256
		key_STRING
		key_UINT96
		key_UINT64
		key_UINT32
		key_DOUBLE
	LIBRARY DESTINATION lib)
