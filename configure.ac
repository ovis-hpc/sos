AC_PREREQ(2.63)
AC_INIT([sosdb], 4.3.4, tom@ogc.us)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([foreign tar-pax])
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

distdir=${PACKAGE_NAME}-${PACKAGE_VERSION}
AC_SUBST(ac_configure_args)
AC_SUBST(prefix)
AC_SUBST(distdir)

AC_SUBST([SOS_SUBST_RULE],["sed \
-e 's,[[@]]mandir[[@]],\$(mandir),g' \
-e 's,[[@]]localedir[[@]],\$(localedir),g' \
-e 's,[[@]]libdir[[@]],\$(libdir),g' \
-e 's,[[@]]psdir[[@]],\$(psdir),g' \
-e 's,[[@]]pdfdir[[@]],\$(pdfdir),g' \
-e 's,[[@]]dvidir[[@]],\$(dvidir),g' \
-e 's,[[@]]htmldir[[@]],\$(htmldir),g' \
-e 's,[[@]]infodir[[@]],\$(infodir),g' \
-e 's,[[@]]docdir[[@]],\$(docdir),g' \
-e 's,[[@]]includedir[[@]],\$(includedir),g' \
-e 's,[[@]]localstatedir[[@]],\$(localstatedir),g' \
-e 's,[[@]]sharedstatedir[[@]],\$(sharedstatedir),g' \
-e 's,[[@]]sysconfdir[[@]],\$(sysconfdir),g' \
-e 's,[[@]]datadir[[@]],\$(datadir),g' \
-e 's,[[@]]datarootdir[[@]],\$(datarootdir),g' \
-e 's,[[@]]libexecdir[[@]],\$(libexecdir),g' \
-e 's,[[@]]sbindir[[@]],\$(sbindir),g' \
-e 's,[[@]]bindir[[@]],\$(bindir),g' \
-e 's,[[@]]prefix[[@]],\$(prefix),g' \
-e 's,[[@]]exec_prefix[[@]],\$(exec_prefix),g' \
-e 's,[[@]]pkglibdir[[@]],\$(pkglibdir),g' \
-e 's,[[@]]PYTHON[[@]],\$(PYTHON),g' \
-e 's,[[@]]pythondir[[@]],\$(pythondir),g' \
"
])

AC_CONFIG_FILES([
		Makefile
		doc/Doxyfile
		doc/Makefile
		man/Makefile
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
