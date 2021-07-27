CFLAGS := $(filter-out -Werror, ${CFLAGS})
PY_INCLUDES = \
	   -I$(top_srcdir)/rpc \
	   -I$(top_srcdir)/sos/include \
	   -I$(top_srcdir)/ods/include \
	   -I$(NUMPY_INCLUDE_PATH)

pkgpyexec_LTLIBRARIES = DSos.la

DSos_la_SOURCES = DSos.pyx DSos.pxd
nodist_DSos_la_SOURCES = DSos.c
DSos_la_CFLAGS = -g -O0  $(PY_INCLUDES) $(PYTHON_CPPFLAGS)
DSos_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
DSos_la_LIBADD = $(top_builddir)/rpc/libdsos.la

pkgpyexecdir = $(pkgpythondir)

# dist_bin_SCRIPTS = dsos-db dsos-part dsos-schema dsos-import-csv dsos-monitor

DSos.c: DSos.pyx DSos.pxd
	echo PYTHON_LDFLAGS are "$(PYTHON_LDFLAGS)"
	cython -3 --gdb --directive language_level=3 \
		-I $(top_srcdir)/sos/python $< -o $@

clean-local:
	rm -f DSos.c
