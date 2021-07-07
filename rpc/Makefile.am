SUBDIRS =
lib_LTLIBRARIES =
sbin_PROGRAMS =
bin_PROGRAMS =

AM_CFLAGS = -Wno-unused-variable -I$(srcdir)/../include -I$(top_srcdir)/ods/include -I$(top_srcdir)/sos/include

libdsos_la_SOURCES = sosapi_client.c sosapi_server.c sosapi_common.c ../ods/src/ods_rbt.c sosapi_svc.c sosapi_xdr.c sosapi_clnt.c
libdsos_la_LIBADD = ../ods/src/libods.la ../sos/src/libsos.la -ldl -lpthread -luuid
libdsos_la_CFLAGS = $(AM_CFLAGS)
lib_LTLIBRARIES += libdsos.la

dsosd_SOURCES = sosapi_server.c ../ods/src/ods_rbt.c ast.c sosapi.h ast.h ../ods/include/ods/ods_rbt.h
dsosd_CFLAGS = $(AM_CFLAGS)
dsosd_LDADD = libdsos.la -lpthread -luuid
bin_PROGRAMS += dsosd

dsosql_SOURCES = dsosql.c dsosql_commands.c ../ods/src/ods_rbt.c ast.c sosapi.h ast.h ../ods/include/ods/ods_rbt.h
dsosql_CFLAGS = $(AM_CFLAGS)
dsosql_LDADD = libdsos.la libjson.la -lpthread -luuid -lreadline
bin_PROGRAMS += dsosql

BUILT_SOURCES = json_parser.h
AM_YFLAGS = -d

EXTRA_DIST = json_lexer.l json_parser.y
CLEANFILES = json_parser.h json_parser.c json_lexer.c

json_parser.c: json_parser.h json.h

json_parser.h: json_parser.y
	$(BISON) --defines=json_parser.h -o json_parser.c $<

json_lexer.c: json_lexer.l json.h json_parser.h
	$(LEX) -o $@ $<

nodist_libjson_la_SOURCES = json_lexer.c json_parser.c json_parser.h
libjson_la_SOURCES = json.c json.h
libjson_la_LIBADD = -lc -lcrypto
lib_LTLIBRARIES += libjson.la
