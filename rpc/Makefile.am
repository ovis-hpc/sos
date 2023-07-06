AUTOMAKE_OPTIONS=subdir-objects
SUBDIRS = . # python
EXTRA_DIST=
lib_LTLIBRARIES =
sbin_PROGRAMS =
bin_PROGRAMS =

BUILT_SOURCES = sosapi.h sosapi_svc.c sosapi_clnt.c sosapi_xdr.c
noinst_HEADERS = sosapi_common.h sosapi.x dsosql.h
nobase_include_HEADERS = dsos.h

xdr-files : FORCE
	rm -f $(srcdir)/sosapi.h
FORCE :

$(BUILT_SOURCES) : sosapi.x

sosapi.h : sosapi.x xdr-files
	rpcgen -N -M $(srcdir)/sosapi.x

AM_CFLAGS = -Wno-unused-variable \
	-I$(top_srcdir)/libtirpc/tirpc \
	-I$(srcdir)/../include \
	-I$(top_srcdir)/ods/include \
	-I$(top_srcdir)/sos/include \
	-I$(top_srcdir)/sos/src

libdsos_la_SOURCES = sosapi_client.c sosapi_server.c sosapi_common.c \
	../ods/src/ods_rbt.c sosapi_svc.c sosapi_xdr.c sosapi_clnt.c \
	ast.c ast.h

libdsos_la_LIBADD = ../ods/src/libods.la ../sos/src/libsos.la \
	../libtirpc/src/libtirpc.la \
	-ldl -lpthread -luuid -lm
libdsos_la_CFLAGS = $(AM_CFLAGS)
lib_LTLIBRARIES += libdsos.la

dsosd_SOURCES = sosapi_server.c ../ods/src/ods_rbt.c ast.c \
	sosapi.h ast.h \
	../ods/include/ods/ods_rbt.h
dsosd_CFLAGS = $(AM_CFLAGS)
dsosd_LDADD = ../sos/src/libsos.la libdsos.la -lpthread -luuid -lm ../libtirpc/src/libtirpc.la -ljansson
bin_PROGRAMS += dsosd

dsosql_SOURCES = dsosql.c dsosql_commands.c \
	ast.c ast.h \
	../ods/src/ods_rbt.c ../ods/include/ods/ods_rbt.h

dsosql_CFLAGS = $(AM_CFLAGS) $(READLINE_CFLAGS)
dsosql_LDADD = ../sos/src/libsos.la libdsos.la -lpthread -luuid $(READLINE_LIBS) -lm -ljansson
bin_PROGRAMS += dsosql

AM_YFLAGS = -d

dist_man8_MANS= dsosd.man

