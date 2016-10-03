AC_PREREQ(2.63)
AC_INIT([SOS], 3.3.0, tom@ogc.us)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([foreign])
AM_PROG_LIBTOOL

dnl Checks for programs
AC_PROG_CC

dnl this will add document options. Please see m4/options.m4.
OPTION_DOC

OPTION_DEFAULT_DISABLE([bwx], [ENABLE_BWX])
OPTION_DEFAULT_ENABLE([yaml], [ENABLE_YAML])

if test -z "$ENABLE_YAML_TRUE"; then
dnl yaml stuff
OPTION_WITH([yaml], [YAML])
OLD_CFLAGS=${CFLAGS}
if test ! -z "${YAML_INCDIR}"; then
   CFLAGS="${CFLAGS} -I${YAML_INCDIR}"
fi
AC_CHECK_HEADER(yaml.h, [],
		AC_MSG_ERROR([yaml.h not found. Please specify yaml prefix with
			--with-yaml configuration option.]))
CFLAGS="${OLD_CFLAGS}"
AC_CHECK_LIB(yaml, yaml_parser_load, [],
		AC_MSG_ERROR([libyaml not found. Please specify yaml prefix with
			--with-yaml configuration option.]),
			[-lyaml ${YAML_LIBDIR_FLAG}])
fi

dnl Reset LIBS variable.
LIBS=""

dnl Swig stuff
OPTION_DEFAULT_DISABLE([swig], [ENABLE_SWIG])
if test -z "$ENABLE_SWIG_TRUE"
then
AM_PATH_PYTHON(2.6)
AC_PROG_SWIG
SWIG_PYTHON
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
		sos/swig/Makefile
		sos/python/Makefile
		bwx/Makefile
		bwx/include/Makefile
		bwx/src/Makefile
		bwx/swig/Makefile
		rpm/rhel7.spec
])

AC_OUTPUT
