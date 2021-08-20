AC_PREREQ(2.63)
AC_INIT([sosdb], 5.1.1, tom@ogc.us)
AM_SILENT_RULES([yes])
AC_CONFIG_SRCDIR([libtirpc/src/auth_des.c])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([foreign tar-pax silent-rules])
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
OPTION_DEFAULT_DISABLE([dsos], [ENABLE_DSOS])
if test -z "$ENABLE_DSOS_TRUE"; then
   AC_PROG_LEX
   AS_IF([test "x$LEX" != "xflex"], [AC_MSG_ERROR([You need to install flex to build DSOS])])
   AC_ARG_VAR([BISON], [bison command])
   AC_CHECK_PROG([BISON], [bison -y], [bison -y], [no])
   AS_IF([test "x$BISON" = "xno"], [AC_MSG_ERROR([You need to install bison to build DSOS])])
fi

AC_CHECK_HEADER([gssapi/gssapi.h], [HAVE_GSSAPI_H=yes], [HAVE_GSSAPI_H=no])

AC_ARG_ENABLE(gssapi,
	[AC_HELP_STRING([--disable-gssapi], [Disable GSSAPI support @<:@default=no@:>@])],
      [],[enable_gssapi=yes])
AM_CONDITIONAL(GSS, test "x$enable_gssapi" = xyes)

if test "x$enable_gssapi" = xyes; then
	if test "x$HAVE_GSSAPI_H" = xno; then
		AC_MSG_ERROR([gssapi.h not found. Use --disable-gssapi, or install GSS-API.])
	fi
	AC_CHECK_TOOL([KRB5_CONFIG], [krb5-config], [no])
	if test "x$KRB5_CONFIG" = xno; then
		AC_MSG_ERROR([krb5-config tool not found. Use --disable-gssapi, or install Kerberos.])
	fi
	GSSAPI_CFLAGS=`${KRB5_CONFIG} --cflags gssapi`
	GSSAPI_LIBS=`${KRB5_CONFIG} --libs gssapi`
	AC_SUBST([GSSAPI_CFLAGS])
	AC_SUBST([GSSAPI_LIBS])
fi

AC_ARG_ENABLE(authdes,
	[AC_HELP_STRING([--enable-authdes], [Enable AUTH_DES support @<:@default=no@:>@])],
      [],[enable_authdes=no])
AM_CONDITIONAL(AUTHDES, test "x$enable_authdes" = xyes)
if test "x$enable_authdes" != xno; then
	AC_DEFINE(AUTHDES_SUPPORT, 1, [Define if AUTH_DES is support])
fi

AC_ARG_ENABLE(ipv6,
	[AC_HELP_STRING([--disable-ipv6], [Disable IPv6 support @<:@default=no@:>@])],
	[],[enable_ipv6=yes])
AM_CONDITIONAL(INET6, test "x$disable_ipv6" != xno)
if test "x$enable_ipv6" != xno; then
	AC_DEFINE(INET6, 1, [Define to 1 if IPv6 is available])
fi

dnl this will add document options. Please see m4/options.m4.
OPTION_DOC

OPTION_DEFAULT_DISABLE([debug], [ENABLE_DEBUG])
OPTION_DEFAULT_ENABLE([python], [ENABLE_PYTHON])
if test -z "$ENABLE_PYTHON_TRUE"; then
	dnl check for python interpreter
	test -n "$PYTHON" || export PYTHON=/usr/bin/python3
 	AM_PATH_PYTHON(,,[:])
 	AX_PYTHON_DEVEL([>='3'])
 	NUMPY_INCLUDE_PATH=$($PYTHON -c "import numpy; print(numpy.get_include())")
 	which cython >/dev/null 2>&1 || AC_MSG_ERROR("You need to install cython when building the Python interfaces")
 	test -n "$NUMPY_INCLUDE_PATH" || AC_MSG_ERROR("You need to install numpy when building the Python interfaces.")
 	AC_MSG_RESULT([${NUMPY_INCLUDE_PATH}])
 	AC_SUBST([NUMPY_INCLUDE_PATH])
fi
AM_CONDITIONAL([HAVE_PYTHON], [test "$PYTHON" != :])

distdir=${PACKAGE_NAME}-${PACKAGE_VERSION}
AC_SUBST(ac_configure_args)
AC_SUBST(prefix)
AC_SUBST(distdir)

LT_VERSION_INFO="3:0:0"
AC_SUBST([LT_VERSION_INFO])
AC_CHECK_HEADER([gssapi/gssapi.h], [HAVE_GSSAPI_H=yes], [HAVE_GSSAPI_H=no])

AC_ARG_ENABLE(gssapi,
	[AC_HELP_STRING([--disable-gssapi], [Disable GSSAPI support @<:@default=no@:>@])],
      [],[enable_gssapi=yes])
AM_CONDITIONAL(GSS, test "x$enable_gssapi" = xyes)

if test "x$enable_gssapi" = xyes; then
	if test "x$HAVE_GSSAPI_H" = xno; then
		AC_MSG_ERROR([gssapi.h not found. Use --disable-gssapi, or install GSS-API.])
	fi
	AC_CHECK_TOOL([KRB5_CONFIG], [krb5-config], [no])
	if test "x$KRB5_CONFIG" = xno; then
		AC_MSG_ERROR([krb5-config tool not found. Use --disable-gssapi, or install Kerberos.])
	fi
	GSSAPI_CFLAGS=`${KRB5_CONFIG} --cflags gssapi`
	GSSAPI_LIBS=`${KRB5_CONFIG} --libs gssapi`
	AC_SUBST([GSSAPI_CFLAGS])
	AC_SUBST([GSSAPI_LIBS])
fi

AC_ARG_ENABLE(authdes,
	[AC_HELP_STRING([--enable-authdes], [Enable AUTH_DES support @<:@default=no@:>@])],
      [],[enable_authdes=no])
AM_CONDITIONAL(AUTHDES, test "x$enable_authdes" = xyes)
if test "x$enable_authdes" != xno; then
	AC_DEFINE(AUTHDES_SUPPORT, 1, [Define if AUTH_DES is support])
fi

AC_ARG_ENABLE(ipv6,
	[AC_HELP_STRING([--disable-ipv6], [Disable IPv6 support @<:@default=no@:>@])],
	[],[enable_ipv6=yes])
AM_CONDITIONAL(INET6, test "x$disable_ipv6" != xno)
if test "x$enable_ipv6" != xno; then
	AC_DEFINE(INET6, 1, [Define to 1 if IPv6 is available])
fi

AC_ARG_ENABLE(symvers,
	[AC_HELP_STRING([--disable-symvers], [Disable symbol versioning @<:@default=no@:>@])],
      [],[enable_symvers=yes])
AM_CONDITIONAL(SYMVERS, test "x$enable_symvers" = xyes)

AC_CANONICAL_BUILD
# Check for which host we are on and setup a few things
# specifically based on the host
case $build_os in
     linux*)
         # Do something specific for linux
         LDFLAG_NOUNDEFINED="-Wl,--no-undefined"
         AC_SUBST(LDFLAG_NOUNDEFINED)
         ;;
     *)
         #Default Case
         ;;
esac

AC_PROG_LIBTOOL
AC_HEADER_DIRENT
AC_PREFIX_DEFAULT(/usr)
AC_CHECK_HEADERS([arpa/inet.h fcntl.h libintl.h limits.h locale.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/param.h sys/socket.h sys/time.h syslog.h unistd.h features.h gssapi/gssapi_ext.h])
AC_CHECK_LIB([pthread], [pthread_create])
AC_CHECK_FUNCS([getrpcbyname getrpcbynumber setrpcent endrpcent getrpcent])

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
		rpcgen/Makefile
		libtirpc/Makefile
		libtirpc/src/Makefile
		libtirpc/man/Makefile
		libtirpc/doc/Makefile
		rpc/Makefile
		rpc/python/Makefile
		rpm/sosdb.spec
])

AC_OUTPUT(libtirpc/libtirpc.pc)
AC_OUTPUT
