cmake_minimum_required(VERSION 2.6)

include_directories(
	"${PROJECT_BINARY_DIR}/../../"
	"${PROJECT_SOURCE_DIR}/include"
)

add_library(sos ods_idx.c ods.c rbt.c)
add_library(idx_BPTREE bpt.c)
# add_library(idx_OBPTREE obpt.c)
add_library(key_BLKMAP key_blkmap.c)
add_library(key_SHA256 key_sha256.c)
add_library(key_STRING key_str.c)
add_library(key_UINT64 key_uint64.c)
add_library(key_UINT32 key_uint32.c)

target_link_libraries(sos dl pthread)
target_link_libraries(idx_BPTREE sos)

add_executable(bpt_test bpt_test.c)
target_link_libraries(bpt_test sos)
