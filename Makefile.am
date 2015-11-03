ACLOCAL_AMFLAGS = -I m4
SUBDIRS = ods sos

if ENABLE_DOC
SUBDIRS += doc
endif

if ENABLE_BWX
SUBDIRS += bwx
endif

# Make rule for RPM on RHEL7
rpm7: dist-gzip
	mkdir -p $@/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
	cp $(distdir).tar.gz $@/SOURCES
	cp rpm/rhel7.spec $@/SPECS
	QA_RPATHS=0x0003 ; \
	export QA_RPATHS ; \
	rpmbuild --define "_topdir `pwd`/$@" \
		--buildroot `pwd`/$@/BUILDROOT \
		-ba $@/SPECS/rhel7.spec

rpm-clean:
	rm -rf rpm7/
	rm -f $(distdir).tar.gz

EXTRA_DIST = autogen.sh COPYING
