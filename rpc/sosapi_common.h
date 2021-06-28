#ifndef _SOSAPI_COMMON_H
#define _SOSAPI_COMMON_H
#include <sos/sos.h>
#include <ods/ods_rbt.h>

/* DSOS errors start at 512, below that they are errno */
enum dsos_error {
	DSOS_ERR_OK = 0,
	DSOS_ERR_MEMORY = 512,
	DSOS_ERR_CLIENT,
	DSOS_ERR_SCHEMA,
	DSOS_ERR_ATTR,
	DSOS_ERR_ITER,
	DSOS_ERR_ITER_EMPTY,
	DSOS_ERR_QUERY_ID,
	DSOS_ERR_QUERY_EMPTY,
	DSOS_ERR_QUERY_BAD_SELECT,
	DSOS_ERR_PARAMETER,
	DSOS_ERR_TRANSPORT
};

#define RPC_ERROR(_err_)	(_err_ + 1024)

typedef struct dsos_result_s {
	int count;
	enum dsos_error any_err;	/* If 0, all return values were 0 */
	enum dsos_error res[16];	/* Response from each server */
} dsos_res_t;

extern sos_schema_t dsos_schema_from_spec(struct dsos_schema_spec *spec);
extern struct dsos_schema_spec * dsos_spec_from_schema(sos_schema_t schema);
extern struct dsos_schema_spec *dsos_schema_spec_dup(dsos_schema_spec *src_spec);
extern void dsos_spec_free(struct dsos_schema_spec *spec);
#endif