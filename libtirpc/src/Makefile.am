## Process this file with automake to create Makefile.in.

## NOTE: this file doesn't really try to be complete.  In particular
## `make dist' won't work at all.  We're just aiming to get the
## program built.  We also don't bother trying to assemble code, or
## anything like that.

CFLAGS := $(filter-out -Werror, ${CFLAGS})
noinst_HEADERS = rpc_com.h debug.h clnt_fd_locks.h nis.h libtirpc.map

AM_CPPFLAGS = -I$(top_srcdir)/libtirpc/tirpc -include config.h -DPORTMAP -DINET6 \
		-D_GNU_SOURCE -Wall -pipe

lib_LTLIBRARIES = libdsosrpc.la

libdsosrpc_la_LDFLAGS = @LDFLAG_NOUNDEFINED@ -no-undefined -lpthread
libdsosrpc_la_LDFLAGS += -version-info @LT_VERSION_INFO@

libdsosrpc_la_SOURCES = auth_none.c auth_unix.c authunix_prot.c \
        binddynport.c bindresvport.c \
        clnt_bcast.c clnt_dg.c clnt_generic.c clnt_perror.c clnt_raw.c clnt_simple.c \
        clnt_vc.c rpc_dtablesize.c getnetconfig.c getnetpath.c getrpcent.c \
        getrpcport.c mt_misc.c pmap_clnt.c pmap_getmaps.c pmap_getport.c \
        pmap_prot.c pmap_prot2.c pmap_rmt.c rpc_prot.c rpc_commondata.c \
        rpc_callmsg.c rpc_generic.c rpc_soc.c rpcb_clnt.c rpcb_prot.c \
        rpcb_st_xdr.c svc.c svc_auth.c svc_dg.c svc_auth_unix.c svc_auth_none.c \
        svc_generic.c svc_raw.c svc_run.c svc_simple.c svc_vc.c getpeereid.c \
        auth_time.c debug.c 

if AUTHDES
libdsosrpc_la_SOURCES += auth_des.c  authdes_prot.c  des_crypt.c  des_impl.c  des_soft.c  svc_auth_des.c
endif

## XDR
libdsosrpc_la_SOURCES += xdr.c xdr_rec.c xdr_array.c xdr_float.c xdr_mem.c xdr_reference.c xdr_stdio.c xdr_sizeof.c

if SYMVERS
    libdsosrpc_la_LDFLAGS += -Wl,--version-script=$(srcdir)/libtirpc.map
endif

## Secure-RPC
if GSS
    libdsosrpc_la_SOURCES += auth_gss.c authgss_prot.c svc_auth_gss.c \
			   rpc_gss_utils.c
    libdsosrpc_la_LIBADD = $(GSSAPI_LIBS)
    libdsosrpc_la_CFLAGS = -DHAVE_RPCSEC_GSS $(GSSAPI_CFLAGS)
endif

libdsosrpc_la_SOURCES += key_call.c key_prot_xdr.c getpublickey.c
libdsosrpc_la_SOURCES += netname.c netnamer.c rpcdname.c rtime.c

CLEANFILES	       = cscope.* *~
DISTCLEANFILES	       = Makefile.in
