CFLAGS := $(filter-out -Werror, ${CFLAGS})
PY_INCLUDES = -I$(srcdir)/../include \
	   -I$(srcdir)/../src \
	   -I$(top_srcdir)/ods/include \
	   -I$(NUMPY_INCLUDE_PATH)

pkgpyexec_LTLIBRARIES = Sos.la

Sos_la_SOURCES = Sos.pyx Sos.pxd
nodist_Sos_la_SOURCES = Sos.c
Sos_la_CFLAGS = -g -O0  $(PY_INCLUDES) $(PYTHON_CPPFLAGS)
Sos_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Sos_la_LIBADD = $(top_builddir)/sos/src/libsos.la

pkgpyexecdir = $(pkgpythondir)

pkgpython_PYTHON = __init__.py DataSet.py

dist_bin_SCRIPTS = sos-db sos-part sos-schema sos-import-csv sos-monitor

Sos.c: Sos.pyx Sos.pxd
	echo PYTHON_LDFLAGS are "$(PYTHON_LDFLAGS)"
	cython -3 --directive language_level=3 $< -o $@

clean-local:
	rm -f Sos.c
