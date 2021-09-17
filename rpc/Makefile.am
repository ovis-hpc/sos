AUTOMAKE_OPTIONS=subdir-objects
SUBDIRS = . # python
lib_LTLIBRARIES =
sbin_PROGRAMS =
bin_PROGRAMS =

noinst_HEADERS = json.h sosapi_common.h sosapi.x dsosql.h json_parser.c json_lexer.c
nobase_include_HEADERS = dsos.h

BUILT_SOURCES = sosapi.h sosapi_svc.c sosapi_clnt.c sosapi_xdr.c
sosapi.h sosapi_svc.c sosapi_clnt.c sosapi_xdr.c &: sosapi.x
	rpcgen -N -M $(srcdir)/sosapi.x

json_parser.c json_parser.h: json_parser.y json.h
	$(BISON) --defines=json_parser.h -o json_parser.c $<

json_lexer.c: json_lexer.l json.h json_parser.h
	$(LEX) -o $@ $<

AM_CFLAGS = -Wno-unused-variable \
	-I$(top_srcdir)/libtirpc/tirpc \
	-I$(srcdir)/../include \
	-I$(top_srcdir)/ods/include \
	-I$(top_srcdir)/sos/include \
	-I$(top_srcdir)/sos/src

libdsos_la_SOURCES = sosapi_client.c sosapi_server.c sosapi_common.c \
	../ods/src/ods_rbt.c sosapi_svc.c sosapi_xdr.c sosapi_clnt.c \
	json_parser.c json_lexer.c json.c \
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
dsosd_LDADD = libdsos.la -lpthread -luuid -lm
bin_PROGRAMS += dsosd

dsosql_SOURCES = dsosql.c dsosql_commands.c \
	json_parser.c json_lexer.c json.c \
	ast.c ast.h \
	../ods/src/ods_rbt.c ../ods/include/ods/ods_rbt.h

dsosql_CFLAGS = $(AM_CFLAGS)
dsosql_LDADD = libdsos.la -lpthread -luuid -lreadline -lm
bin_PROGRAMS += dsosql

BUILT_SOURCES = json_parser.h json_parser.c json_lexer.c
AM_YFLAGS = -d

EXTRA_DIST = json_lexer.l json_parser.y
CLEANFILES = json_parser.h json_parser.c json_lexer.c
