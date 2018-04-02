lib_LTLIBRARIES =

# If xxx_CFLAGS is not specified, AM_CFLAGS will be used.
AM_CFLAGS = -I$(srcdir)/../include/

if ENABLE_DEBUG
AM_CFLAGS += -DODS_DEBUG
endif

ods_dump_SOURCES = ods_dump.c
ods_dump_CFLAGS = $(AM_CFLAGS)
ods_dump_LDADD = libods.la
bin_PROGRAMS = ods_dump

libods_la_SOURCES = ods_idx.c ods.c ods_opt.c rbt.c ods_log.c ods_idx_priv.h ods_priv.h oidx_priv.h fnv_hash.h
libods_la_LIBADD = -ldl -lpthread
# libods_la_LDFLAGS = -pg
lib_LTLIBRARIES += libods.la

libidx_HTBL_la_SOURCES = ht.c ht.h
# libidx_HTBL_la_CFLAGS = $(AM_CFLAGS)
libidx_HTBL_la_CFLAGS = $(AM_CFLAGS) -DHT_THREAD_SAFE
# libidx_BXTREE_la_LDFLAGS = -pg
libidx_HTBL_la_LIBADD = libods.la
lib_LTLIBRARIES += libidx_HTBL.la

libidx_BXTREE_la_SOURCES = bxt.c bxt.h
# libidx_BXTREE_la_CFLAGS = $(AM_CFLAGS)
libidx_BXTREE_la_CFLAGS = $(AM_CFLAGS)
# libidx_BXTREE_la_LDFLAGS = -pg
libidx_BXTREE_la_LIBADD = libods.la
lib_LTLIBRARIES += libidx_BXTREE.la

libidx_H2BXT_la_SOURCES = h2bxt.c h2bxt.h mq.c mq.h
#libidx_H2BXT_la_CFLAGS = $(AM_CFLAGS)
libidx_H2BXT_la_CFLAGS = $(AM_CFLAGS)
# libidx_H2BXT_la_LDFLAGS = -pg
libidx_H2BXT_la_LIBADD = libods.la
lib_LTLIBRARIES += libidx_H2BXT.la

libkey_COMPOUND_la_SOURCES = key_compound.c
lib_LTLIBRARIES += libkey_COMPOUND.la

libkey_BLKMAP_la_SOURCES = key_blkmap.c
lib_LTLIBRARIES += libkey_BLKMAP.la

libkey_SHA256_la_SOURCES = key_sha256.c
lib_LTLIBRARIES += libkey_SHA256.la

libkey_STRING_la_SOURCES = key_str.c
lib_LTLIBRARIES += libkey_STRING.la

libkey_MEMCMP_la_SOURCES = key_memcmp.c
lib_LTLIBRARIES += libkey_MEMCMP.la

libkey_UINT192_la_SOURCES = key_uint192.c
lib_LTLIBRARIES += libkey_UINT192.la

libkey_UINT128_la_SOURCES = key_uint128.c
lib_LTLIBRARIES += libkey_UINT128.la

libkey_UINT96_la_SOURCES = key_uint96.c
lib_LTLIBRARIES += libkey_UINT96.la

libkey_UINT64_la_SOURCES = key_uint64.c
lib_LTLIBRARIES += libkey_UINT64.la

libkey_INT64_la_SOURCES = key_int64.c
lib_LTLIBRARIES += libkey_INT64.la

libkey_UINT32_la_SOURCES = key_uint32.c
lib_LTLIBRARIES += libkey_UINT32.la

libkey_INT32_la_SOURCES = key_int32.c
lib_LTLIBRARIES += libkey_INT32.la

libkey_UINT16_la_SOURCES = key_uint16.c
lib_LTLIBRARIES += libkey_UINT16.la

libkey_INT16_la_SOURCES = key_int16.c
lib_LTLIBRARIES += libkey_INT16.la

libkey_DOUBLE_la_SOURCES = key_double.c
lib_LTLIBRARIES += libkey_DOUBLE.la
