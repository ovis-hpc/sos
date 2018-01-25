AC_PREREQ(2.63)
AC_INIT([sosdb], 4.1.0, tom@ogc.us)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([foreign])
AM_PROG_LIBTOOL

dnl Checks for programs
AC_PROG_CC

dnl this will add document options. Please see m4/options.m4.
OPTION_DOC

OPTION_DEFAULT_DISABLE([debug], [ENABLE_DEBUG])
OPTION_DEFAULT_ENABLE([python], [ENABLE_PYTHON])

if test -z "$ENABLE_PYTHON_TRUE"
then
  AM_PATH_PYTHON(,,[:])
  NUMPY_INCLUDE_PATH=$($PYTHON -c "import numpy; print(numpy.get_include())")
  which cython >/dev/null 2>&1 || AC_MSG_ERROR("cython not found")
  test -n "$NUMPY_INCLUDE_PATH" || AC_MSG_ERROR("numpy not found.")
  AX_PYTHON_DEVEL()
  AC_MSG_RESULT([${NUMPY_INCLUDE_PATH}])
  AC_SUBST([NUMPY_INCLUDE_PATH])
fi

distdir=${PACKAGE_NAME}-${PACKAGE_VERSION}
AC_SUBST(ac_configure_args)
AC_SUBST(prefix)
AC_SUBST(distdir)

AC_CONFIG_FILES([
		Makefile
		doc/Doxyfile
		doc/Makefile
		ods/Makefile
		ods/include/Makefile
		ods/src/Makefile
		sos/Makefile
		sos/include/Makefile
		sos/src/Makefile
		sos/python/Makefile
		rpm/sosdb.spec
])

AC_OUTPUT
