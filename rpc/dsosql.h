#ifndef __DSOSQL_H__
#define __DSOSQL_H__
#include "dsos.h"
int dsosql_import_csv(dsos_container_t cont, FILE* fp, char *schema_name, char *col_spec);
int dsosql_create_schema(dsos_container_t cont, char *schema_name, char *template);
int dsosql_query_select(dsos_container_t cont, const char *select_clause);
struct col_s {
	const char *name;
	int id;
	int width;
	TAILQ_ENTRY(col_s) entry;
};
extern TAILQ_HEAD(col_list_s, col_s) col_list;
int add_column(sos_schema_t schema, const char *str);
void table_header(FILE *outp, sos_attr_t attr);
void table_footer(FILE *outp, int rec_count, int iter_count);
void table_row(FILE *outp, sos_schema_t schema, sos_obj_t obj);
#endif
