CFLAGS := $(filter-out -Werror, ${CFLAGS})
PY_INCLUDES = -I$(srcdir)/../include \
	   -I$(srcdir)/../src \
	   -I$(top_srcdir)/ods/include \
	   -I$(NUMPY_INCLUDE_PATH)

pkgpyexec_LTLIBRARIES = Sos.la Ldms.la Array.la

Sos_la_SOURCES = Sos.pyx Sos.pxd
nodist_Sos_la_SOURCES = Sos.c
Sos_la_CFLAGS = -g -O0  $(PY_INCLUDES) $(PYTHON_CPPFLAGS)
Sos_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Sos_la_LIBADD = $(top_builddir)/sos/src/libsos.la

# Array_la_SOURCES = Array.pyx Array.pxd
Array_la_SOURCES = Array.pyx
nodist_Array_la_SOURCES = Array.c
Array_la_CFLAGS = -g -O0  $(PY_INCLUDES) $(PYTHON_CPPFLAGS)
Array_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Array_la_LIBADD = $(top_builddir)/sos/src/libsos.la

Ldms_la_SOURCES = Ldms.pyx
nodist_Ldms_la_SOURCES = Ldms.c
Ldms_la_CFLAGS = -g -O0  $(PY_INCLUDES) $(PYTHON_CPPFLAGS)
Ldms_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Ldms_la_LIBADD = $(top_builddir)/sos/src/libsos.la

pkgpyexecdir = $(pkgpythondir)

pkgpython_PYTHON = __init__.py

dist_bin_SCRIPTS = sos-db sos-part sos-schema sos-import-csv sos-monitor lmq

Sos.c: Sos.pyx Sos.pxd
	echo PYTHON_LDFLAGS are "$(PYTHON_LDFLAGS)"
	cython --gdb $< -o $@

Ldms.c: Ldms.pyx Sos.pxd
	cython --gdb $< -o $@

Array.c : Array.pyx
	cython --gdb $< -o $@

clean-local:
	rm -f Sos.c
	rm -f Ldms.c
	rm -f Array.c
