#ifndef _DSOS_H_
#define _DSOS_H_
#include <inttypes.h>
#include <time.h>
#include <sos/sos.h>

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

static inline double dsos_diff_timespec_s(struct timespec start, struct timespec end)
{
	uint64_t nsecs_start, nsecs_end;

	nsecs_start = start.tv_sec * 1000000000 + start.tv_nsec;
	nsecs_end = end.tv_sec * 1000000000 + end.tv_nsec;
	return (double)(nsecs_end - nsecs_start) / 1e9;
}

typedef struct dsos_result_s {
	int count;
	enum dsos_error any_err;	/* If 0, all return values were 0 */
	enum dsos_error res[16];	/* Response from each server */
} dsos_res_t;

typedef struct dsos_session_s *dsos_session_t;
typedef struct dsos_container_s *dsos_container_t;
typedef struct dsos_schema_s *dsos_schema_t;
typedef struct dsos_part_s *dsos_part_t;
typedef struct dsos_iter_s *dsos_iter_t;
typedef struct dsos_query_s *dsos_query_t;

typedef struct dsos_name_array_s {
	int count;
	char **names;
} *dsos_name_array_t;

extern void dsos_session_close(dsos_session_t sess);
extern dsos_session_t dsos_session_open(const char *config_file);
extern void dsos_container_close(dsos_container_t cont);
extern void dsos_container_commit(dsos_container_t cont);
extern dsos_container_t dsos_container_open(dsos_session_t sess, const char *path, sos_perm_t perm, int mode);
extern int dsos_container_error(dsos_container_t cont, char **msg);
extern dsos_schema_t dsos_schema_create(dsos_container_t cont, sos_schema_t schema, dsos_res_t *res);
extern void dsos_schema_free(dsos_schema_t schema);
extern dsos_schema_t dsos_schema_by_name(dsos_container_t cont, const char *name);
extern dsos_schema_t dsos_schema_by_uuid(dsos_container_t cont, const uuid_t uuid);
extern dsos_name_array_t dsos_schema_query(dsos_container_t cont);
extern sos_schema_t dsos_schema_local(dsos_schema_t schema);
extern int dsos_schema_attr_count(dsos_schema_t schema);
extern sos_attr_t dsos_schema_attr_by_id(dsos_schema_t schema, int attr_id);
extern sos_attr_t dsos_schema_attr_by_name(dsos_schema_t schema, const char *name);
extern void dsos_schema_print(dsos_schema_t schema, FILE *fp);
extern dsos_part_t dsos_part_create(dsos_container_t cont,
			const char *name, const char *path, const char *desc,
			uid_t uid, gid_t gid, int perm);
void dsos_part_free(dsos_part_t part);
extern dsos_part_t dsos_part_by_name(dsos_container_t cont, const char *name);
extern dsos_part_t dsos_part_by_uuid(dsos_container_t cont, const uuid_t uuid);
extern const char *dsos_part_name(dsos_part_t part);
extern const char *dsos_part_desc(dsos_part_t part);
extern const char *dsos_part_path(dsos_part_t part);
extern sos_part_state_t dsos_part_state(dsos_part_t part);
extern void dsos_part_uuid(dsos_part_t part, uuid_t uuid);
extern uid_t dsos_part_uid(dsos_part_t part);
extern gid_t dsos_part_gid(dsos_part_t part);
extern int dsos_part_perm(dsos_part_t part);
extern dsos_name_array_t dsos_part_query(dsos_container_t cont);
extern int dsos_part_chown(dsos_part_t part, uid_t uid, gid_t gid);
extern int dsos_part_chmod(dsos_part_t part, int perm);

extern int dsos_transaction_begin(dsos_container_t cont, struct timespec *timeout);
extern int dsos_transaction_end(dsos_container_t cont);
extern int dsos_obj_create(dsos_container_t cont, dsos_part_t part,
	dsos_schema_t schema, sos_obj_t obj);
extern int dsos_obj_update(dsos_container_t cont, sos_obj_t obj);
extern int dsos_obj_delete(dsos_container_t cont, sos_obj_t obj);
extern sos_obj_t dsos_obj_new(dsos_schema_t schema);
extern sos_obj_t dsos_obj_new_size(dsos_schema_t schema, size_t reserve);
extern dsos_iter_t dsos_iter_create(dsos_container_t cont, dsos_schema_t schema, const char *attr_name);
extern void dsos_iter_delete(dsos_iter_t iter);
extern sos_obj_t dsos_iter_obj(dsos_iter_t iter);
extern sos_key_t dsos_iter_key(dsos_iter_t iter, sos_obj_t obj);
extern sos_attr_t dsos_iter_attr(dsos_iter_t iter);
extern dsos_schema_t dsos_iter_schema(dsos_iter_t iter);
extern sos_obj_t dsos_iter_begin(dsos_iter_t iter);
extern sos_obj_t dsos_iter_end(dsos_iter_t iter);
extern sos_obj_t dsos_iter_next(dsos_iter_t iter);
extern sos_obj_t dsos_iter_prev(dsos_iter_t iter);

extern sos_obj_t dsos_iter_find_glb(dsos_iter_t iter, sos_key_t key);
extern sos_obj_t dsos_iter_find_lub(dsos_iter_t iter, sos_key_t key);
extern sos_obj_t dsos_iter_find_le(dsos_iter_t iter, sos_key_t key);
extern sos_obj_t dsos_iter_find_ge(dsos_iter_t iter, sos_key_t key);
extern sos_obj_t dsos_iter_find(dsos_iter_t iter, sos_key_t key);

extern sos_obj_t dsos_index_find(dsos_container_t cont, dsos_schema_t dschema,
				 sos_attr_t attr, sos_key_t key);
extern sos_obj_t dsos_index_find_le(dsos_container_t cont, dsos_schema_t dschema,
				 sos_attr_t attr, sos_key_t key);
extern sos_obj_t dsos_index_find_ge(dsos_container_t cont, dsos_schema_t dschema,
				 sos_attr_t attr, sos_key_t key);

typedef struct dsos_iter_stats_s {
	uint64_t cardinality;
	uint64_t duplicates;
	uint64_t size_bytes;
} dsos_iter_stats_t;
extern dsos_iter_stats_t dsos_iter_stats(dsos_iter_t iter);
extern int dsos_attr_value_min(dsos_container_t cont, sos_attr_t attr);
extern int dsos_attr_value_max(dsos_container_t cont, sos_attr_t attr);
extern dsos_query_t dsos_query_create(dsos_container_t cont);
extern void dsos_query_destroy(dsos_query_t query);
extern int dsos_query_select(dsos_query_t query, const char *clause);
extern sos_obj_t dsos_query_next(dsos_query_t query);
extern sos_schema_t dsos_query_schema(dsos_query_t query);
extern sos_attr_t dsos_query_index_attr(dsos_query_t query);
extern const char *dsos_query_errmsg(dsos_query_t query);
extern const int dsos_last_err(void);
extern const char *dsos_last_errmsg(void);
extern void dsos_name_array_free(dsos_name_array_t names);
#endif
