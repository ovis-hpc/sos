lib_LTLIBRARIES =
sbin_PROGRAMS =
bin_PROGRAMS =

BUILT_SOURCES = json_parser.h json_parser.c json_lexer.c
noinst_HEADERS = json.h

AM_CFLAGS = -Wno-unused-variable \
	-I$(top_srcdir)/ods/include

json_parser.c json_parser.h : json_parser.y json.h
	$(BISON) --defines=json_parser.h -o json_parser.c $<

json_lexer.c: json_lexer.l json.h json_parser.h
	$(LEX) -o $@ $<

libsos_json_la_SOURCES = json_parser.c json_lexer.c json.c
libsos_json_la_CFLAGS = $(AM_CFLAGS)
lib_LTLIBRARIES += libsos_json.la

EXTRA_DIST = json_lexer.l json_parser.y
CLEANFILES = json_parser.h json_parser.c json_lexer.c
