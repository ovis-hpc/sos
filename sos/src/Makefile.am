lib_LTLIBRARIES =
bin_PROGRAMS =

AM_CFLAGS = -I$(srcdir)/../include -I$(top_srcdir)/ods/include -I$(top_srcdir)/sos/include
if ENABLE_DEBUG
AM_CFLAGS += -DODS_DEBUG
endif

libsos_la_SOURCES = sos_fn.c \
		    sos.c \
		    sos_part.c \
		    sos_schema.c \
		    sos_config.c \
		    sos_index.c \
		    sos_key.c \
		    sos_iter.c \
		    sos_value.c \
		    sos_log.c \
		    sos_priv.h

libsos_la_LIBADD = ../../ods/src/libods.la -ldl -lpthread
lib_LTLIBRARIES += libsos.la

sos_cmd_SOURCES = sos_cmd.c
sos_cmd_CFLAGS = $(AM_CFLAGS)
sos_cmd_LDADD = libsos.la -lpthread
bin_PROGRAMS += sos_cmd

sos_part_create_SOURCES = sos_part_create.c
sos_part_create_LDADD = libsos.la
bin_PROGRAMS += sos_part_create

sos_part_query_SOURCES = sos_part_query.c
sos_part_query_LDADD = libsos.la
bin_PROGRAMS += sos_part_query

sos_part_modify_SOURCES = sos_part_modify.c
sos_part_modify_LDADD = libsos.la
bin_PROGRAMS += sos_part_modify

sos_part_delete_SOURCES = sos_part_delete.c
sos_part_delete_LDADD = libsos.la
bin_PROGRAMS += sos_part_delete

sos_part_move_SOURCES = sos_part_move.c
sos_part_move_LDADD = libsos.la
bin_PROGRAMS += sos_part_move

sos_part_export_SOURCES = sos_part_export.c
sos_part_export_LDADD = libsos.la
bin_PROGRAMS += sos_part_export

sos_part_index_SOURCES = sos_part_index.c
sos_part_index_LDADD = libsos.la
bin_PROGRAMS += sos_part_index
