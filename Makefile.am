ACLOCAL_AMFLAGS = -I m4
SUBDIRS = ods sos
SUBDIRS += rpcgen libtirpc rpc
SUBDIRS += sos/python

if ENABLE_PYTHON
SUBDIRS += sos/python
endif


if ENABLE_DOC
SUBDIRS += doc
endif

# Make rule for RPM on RHEL7
sosdb_rpm: dist-gzip
	mkdir -p $@/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
	cp $(distdir).tar.gz $@/SOURCES
	cp rpm/sosdb.spec $@/SPECS
	rpmbuild --define "_topdir `pwd`/$@" \
		--buildroot `pwd`/$@/BUILDROOT \
		-ba $@/SPECS/sosdb.spec

sosdb_rpm-clean:
	rm -rf sosdb_rpm/
	rm -f $(distdir).tar.gz

EXTRA_DIST = autogen.sh COPYING
