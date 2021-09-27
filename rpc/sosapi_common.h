#ifndef _SOSAPI_COMMON_H
#define _SOSAPI_COMMON_H
#include <sos/sos.h>
#include <ods/ods_rbt.h>

#define RPC_ERROR(_err_)	(_err_ + 1024)
#define IS_RPC_ERROR(_err_)	(_err_ >= 1024)

extern sos_schema_t dsos_schema_from_spec(struct dsos_schema_spec *spec);
extern struct dsos_schema_spec * dsos_spec_from_schema(sos_schema_t schema);
extern struct dsos_schema_spec *dsos_schema_spec_dup(dsos_schema_spec *src_spec);
extern void dsos_spec_free(struct dsos_schema_spec *spec);
extern int dsos_part_spec_from_part(dsos_part_spec *spec, sos_part_t part);
extern dsos_part_spec *dsos_part_spec_dup(dsos_part_spec *spec);

#endif
