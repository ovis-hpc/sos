.PHONY: all doc docs
all: doc
doc: docs
docs: doxygen-out

DOCINPUT = $(top_srcdir)/sos/include/sos/sos.h \
		       $(top_srcdir)/ods/include/ods/ods.h \
		       $(top_srcdir)/ods/include/ods/ods_idx.h \
		       $(top_srcdir)/sos/src/sos.c \
		       $(top_srcdir)/sos/src/sos_part.c \
		       $(top_srcdir)/sos/src/sos_config.c \
		       $(top_srcdir)/sos/src/sos_schema.c \
		       $(top_srcdir)/sos/src/sos_index.c \
		       $(top_srcdir)/sos/src/sos_iter.c \
		       $(top_srcdir)/sos/src/sos_value.c \
		       $(top_srcdir)/sos/src/sos_key.c \
		       $(top_srcdir)/sos/src/sos_cmd.c \
		       $(top_srcdir)/sos/src/sos_part_create.c \
		       $(top_srcdir)/sos/src/sos_part_query.c \
		       $(top_srcdir)/sos/src/sos_part_modify.c \
		       $(top_srcdir)/sos/src/sos_part_delete.c \
		       $(top_srcdir)/sos/src/sos_part_move.c


doxygen-out: $(DOCINPUT)
	( cat Doxyfile && echo "INPUT = $(DOCINPUT)" ) | doxygen -
	touch doxygen-out

.PHONY: install-doc-dir
install-doc-dir:
	mkdir -p $(DESTDIR)$(docdir)

.PHONY: install-doc-latex uninstall-doc-latex
if ENABLE_DOC_LATEX
DEST_PDF = $(DESTDIR)$(docdir)/$(PACKAGE)-$(VERSION).pdf
docs: latex/refman.pdf

latex/refman.pdf: doxygen-out
	$(MAKE) -C latex

install-doc-latex: install-doc-dir latex/refman.pdf
	cp latex/refman.pdf $(DEST_PDF)

install-data-local: install-doc-latex

uninstall-doc-latex:
	rm -f $(DEST_PDF)

uninstall-local: uninstall-doc-latex
endif # ENABLE_DOC_LATEX

.PHONY: install-doc-html uninstall-doc-html
if ENABLE_DOC_HTML
install-doc-html: install-doc-dir doxygen-out
	cp -r html $(DESTDIR)$(docdir)/

install-data-local: install-doc-html

uninstall-doc-html:
	rm -rf $(DESTDIR)$(docdir)/html

uninstall-local: uninstall-doc-html
endif # ENABLE_DOC_HTML

.PHONY: install-doc-man uninstall-doc-man

if ENABLE_DOC_MAN
install-doc-man: install-doc-dir doxygen-out
	cp -r man $(DESTDIR)$(docdir)/

install-data-local: install-doc-man

uninstall-doc-man:
	rm -rf $(DESTDIR)$(docdir)/man

uninstall-local: uninstall-doc-man
endif # ENABLE_DOC_MAN

clean-local:
	rm -rf html latex man doxygen-out
