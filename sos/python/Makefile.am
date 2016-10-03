CFLAGS := $(filter-out -Werror, ${CFLAGS})
INCLUDES = -I$(srcdir)/../include \
	   -I$(srcdir)/../src \
	   -I$(top_srcdir)/ods/include \
	   -I/usr/include

pkgpyexec_LTLIBRARIES = Sos.la Ldms.la

Sos_la_SOURCES = Sos.pyx Sos.pxd
nodist_Sos_la_SOURCES = Sos.c
Sos_la_CFLAGS = -g -O0 -I$(top_srcdir) $(SWIG_PYTHON_CPPFLAGS) $(INCLUDES)
Sos_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Sos_la_LIBADD = $(top_builddir)/sos/src/libsos.la

Ldms_la_SOURCES = Ldms.pyx
nodist_Ldms_la_SOURCES = Ldms.c
Ldms_la_CFLAGS = -g -O0 -I$(top_srcdir) $(SWIG_PYTHON_CPPFLAGS) $(INCLUDES)
Ldms_la_LDFLAGS = $(PYTHON_LDFLAGS) -module -shared
Ldms_la_LIBADD = $(top_builddir)/sos/src/libsos.la

pkgpyexecdir = $(pkgpythondir)

Sos.c: Sos.pyx Sos.pxd
	cython --gdb $< -o $@

Ldms.c: Ldms.pyx Sos.pxd
	cython --gdb $< -o $@

clean-local:
	rm -f Sos.c
	rm -f Ldms.c
