AM_CFLAGS = -I$(top_srcdir)/ods/include \
	    -I$(top_srcdir)/sos/include \
	    -I$(srcdir)/../include
bin_PROGRAMS =

bwx_job_add_SOURCES = bwx_job_add.c
bwx_job_add_LDADD = ../../sos/src/libsos.la
bin_PROGRAMS += bwx_job_add

bwx_import_SOURCES = bwx_import.c
bwx_import_LDADD = ../../sos/src/libsos.la -lpthread
bin_PROGRAMS += bwx_import

bwx_job_ls_SOURCES = bwx_job_ls.c
bwx_job_ls_LDADD = ../../sos/src/libsos.la -lm
bin_PROGRAMS += bwx_job_ls
