# For Swig stuffs
AM_CFLAGS = -I$(srcdir)/../include \
	    -I$(srcdir)/../src \
	    -I$(top_srcdir)/ods/include \
	    -I$(top_srcdir)/sos/include \
	    -I/usr/include

BUILT_SOURCES = bwx.c bwx.py
SWIG_SOURCES = bwx.i \
	       $(srcdir)/../include/bwx/bwx.h \
	       $(top_srcdir)/sos/include/sos/sos.h
pkgpythondir="${pythondir}/bwx"
pkgpyexecdir="${pkgpythondir}"
pkgpython_PYTHON = BWX.py __init__.py
nodist_pkgpython_PYTHON = bwx.py
pkgpyexec_LTLIBRARIES = _bwx.la
_bwx_la_SOURCES = $(SWIG_SOURCES)
nodist__bwx_la_SOURCES = bwx.c
_bwx_la_CFLAGS = -g $(AM_CFLAGS) $(SWIG_PYTHON_CPPFLAGS)
_bwx_la_LDFLAGS = $(PYTHON_LDFLAGS) -module
_bwx_la_LIBADD = ../../sos/src/libsos.la

bwx.py: bwx.c

bwx.c: $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) $(AM_CFLAGS) -o $@ $<

clean-local:
	rm -f bwx.c bwx.py
