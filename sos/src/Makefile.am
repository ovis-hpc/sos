lib_LTLIBRARIES =
bin_PROGRAMS =

AM_CFLAGS = -I$(srcdir)/../include -I$(top_srcdir)/ods/include

libsos_la_SOURCES = sos_fn.c \
		    sos.c \
		    sos_schema.c \
		    sos_config.c \
		    sos_index.c \
		    sos_key.c \
		    sos_iter.c \
		    sos_value.c
libsos_la_LIBADD = ../../ods/src/libods.la -ldl -lpthread
lib_LTLIBRARIES += libsos.la

sos_cmd_SOURCES = sos_cmd.c
sos_cmd_CFLAGS = $(AM_CFLAGS) @YAML_INCDIR_FLAG@
sos_cmd_LDADD = libsos.la -lyaml @YAML_LIBDIR_FLAG@ -lpthread
bin_PROGRAMS += sos_cmd

sos_part_SOURCES = sos_part.c
sos_part_LDADD = libsos.la -lpthread
bin_PROGRAMS += sos_part
