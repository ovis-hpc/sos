# For Swig stuffs
AM_CFLAGS = -I$(srcdir)/../include \
	    -I$(srcdir)/../src \
	    -I$(top_srcdir)/ods/include \
	    -I/usr/include/x86_64-linux-gnu \
	    -I/usr/include

BUILT_SOURCES = sos.c sos.py
SWIG_SOURCES = sos.i $(srcdir)/../src/sos_priv.h \
	       $(srcdir)/../include/sos/sos.h
pkgpythondir="${pythondir}/sos"
pkgpyexecdir="${pkgpythondir}"
pkgpython_PYTHON = SOS.py __init__.py
nodist_pkgpython_PYTHON = sos.py
pkgpyexec_LTLIBRARIES = _sos.la
nodist__sos_la_SOURCES = sos.c
_sos_la_SOURCES = $(SWIG_SOURCES)
_sos_la_CFLAGS = -g $(AM_CFLAGS) $(SWIG_PYTHON_CPPFLAGS)
_sos_la_LDFLAGS = $(PYTHON_LDFLAGS) -module
_sos_la_LIBADD = ../src/libsos.la -lpthread

# Forcing this so that _sos.so will be in lib not lib64
#pkgpyexecdir = $(pkgpythondir)
# not a good idea; use correct pythonpath

sos.py: sos.c

sos.c: $(SWIG_SOURCES)
	$(SWIG) $(SWIG_PYTHON_OPT) $(AM_CFLAGS) -o $@ $<

clean-local:
	rm -f sos.c sos.py

# $(PYTHON_SITE_PKG) is set by AC_PYTHON_DEVEL
# $(pkgpyexecdir) and $(pkgpythondir) is set by automake-python
# $(pythondir) and $(pyexecdir) is also set by automake-python

sos_pth = $(PYTHON_SITE_PKG)/sos.pth
# Disable this for non-root installation
#
#install-data-hook:
#	echo "$(pythondir)" > $(sos_pth)

#uninstall-local:
#	rm -f $(sos_pth)

