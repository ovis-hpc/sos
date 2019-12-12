SUBDIRS = src include

if ENABLE_PYTHON
SUBDIRS += python
endif

EXTRA_DIST = sos-configvars.sh.in
CLEANFILES = sos-configvars.sh
do_subst=@SOS_SUBST_RULE@

%.sh: %.sh.in Makefile
	$(do_subst) < $< > $@
	chmod 755 $@

install-data-hook: sos-configvars.sh
	$(MKDIR_P) $(DESTDIR)$(libdir)
	$(INSTALL) -m 755 sos-configvars.sh $(DESTDIR)$(libdir)

uninstall-hook:
	rm $(DESTDIR)/$(libdir)/sos-configvars.sh
