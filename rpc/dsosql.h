#ifndef __DSOSQL_H__
#define __DSOSQL_H__
#include "dsos.h"
int dsosql_import_csv(dsos_container_t cont, FILE* fp, char *schema_name, char *col_spec);
int dsosql_create_schema(dsos_container_t cont, char *schema_name, char *template);
int dsosql_query_select(dsos_container_t cont, const char *select_clause);
#endif
