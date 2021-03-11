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

libsos_la_LIBADD = ../../ods/src/libods.la -ldl -lpthread -luuid
lib_LTLIBRARIES += libsos.la

sos_cmd_SOURCES = sos_cmd.c
sos_cmd_CFLAGS = $(AM_CFLAGS)
sos_cmd_LDADD = libsos.la -lpthread -luuid
bin_PROGRAMS += sos_cmd

sos_cont_clone_SOURCES = sos_cont_clone.c
sos_cont_clone_LDADD = libsos.la
bin_PROGRAMS += sos_cont_clone

sos_part_create_SOURCES = sos_part_create.c
sos_part_create_LDADD = libsos.la
bin_PROGRAMS += sos_part_create

sos_part_attach_SOURCES = sos_part_attach.c
sos_part_attach_LDADD = libsos.la
bin_PROGRAMS += sos_part_attach

sos_part_detach_SOURCES = sos_part_detach.c
sos_part_detach_LDADD = libsos.la
bin_PROGRAMS += sos_part_detach

sos_part_query_SOURCES = sos_part_query.c
sos_part_query_LDADD = libsos.la
bin_PROGRAMS += sos_part_query

sos_part_modify_SOURCES = sos_part_modify.c
sos_part_modify_LDADD = libsos.la
bin_PROGRAMS += sos_part_modify

sos_part_move_SOURCES = sos_part_move.c
sos_part_move_LDADD = libsos.la
bin_PROGRAMS += sos_part_move
