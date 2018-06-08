AC_PREREQ(2.63)
AC_INIT([sosdb], 4.1.0, tom@ogc.us)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([foreign])
AM_PROG_LIBTOOL

COMMIT_ID="$(git rev-parse HEAD 2>/dev/null)"
if test -z "$COMMIT_ID"; then
	dnl Git OK from ovis repo.
	AC_MSG_RESULT([The git command is not available, commit_id is "00..."])
	COMMIT_ID="0000000000000000000000000000000000000000"
fi
AC_DEFINE_UNQUOTED([ODS_COMMIT_ID],["$COMMIT_ID"], [Git commit id])
AC_SUBST([ODS_COMMIT_ID],["$COMMIT_ID"])

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

dnl Check for tcmalloc
OCFLAGS=$CFLAGS
CFLAGS=""
LIB_TCMALLOC=""
AC_CHECK_LIB(tcmalloc,
		malloc,
		[LIB_TCMALLOC="-ltcmalloc"],
		[AC_MSG_WARN(You may wish to install libtcmalloc for better performance)])
AC_SUBST([LIB_TCMALLOC])
CFLAGS=$OCFLAGS

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
