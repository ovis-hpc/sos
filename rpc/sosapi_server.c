/* -*- c-basic-offset: 8 -*- */
#define _GNU_SOURCE
#include <sos/sos.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include <ods/ods_rbt.h>
#include "sosapi.h"
#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <dirent.h>
#include <errno.h>
#include <assert.h>
#include <sys/queue.h>
#include <syslog.h>
#include "dsos.h"
#include "sosapi.h"          /* Created by rpcgen */
#include "sosapi_common.h"
#include "ast.h"

static struct timespec ZERO_TIME = {
	0, 0
};

struct dsos_session {
	uint64_t handle;
	SVCXPRT *xprt;
	struct timespec open_time;
	struct timespec acc_time;
	struct timespec x_start_time;
	struct timespec x_end_time;

	sos_t sos;
	int result;
	pthread_mutex_t schema_tree_lock;
	struct ods_rbt schema_id_tree;
	struct ods_rbt schema_name_tree;
	struct ods_rbt schema_uuid_tree;
	pthread_mutex_t iter_tree_lock;
	struct ods_rbt iter_tree;
	pthread_mutex_t query_tree_lock;
	struct ods_rbt query_tree;
	struct ods_rbn rbn;
};

struct dsos_object {
	uint64_t handle;
	sos_obj_t obj;
	struct ods_rbn rbn;
};

struct dsos_iter {
	uint64_t handle;
	dsos_container_id cont_id;
	dsos_schema_id schema_id;
	sos_iter_t iter;
	struct ods_rbn rbn;
};

struct dsos_schema {
	uint64_t handle;
	sos_schema_t schema;
	dsos_schema_spec *spec;
	struct ods_rbn id_rbn;
	struct ods_rbn name_rbn;
	struct ods_rbn uuid_rbn;
};

struct dsos_query {
	uint64_t handle;
	enum dsos_query_state_e {
		DSOSQ_STATE_INIT = 0,
		DSOSQ_STATE_BEGIN,
		DSOSQ_STATE_NEXT,
		DSOSQ_STATE_EMPTY
	} state;
	dsos_container_id cont_id;
	struct ast *ast;
	struct ods_rbn rbn;
};

static uint64_t next_handle = 1;
static inline uint64_t get_next_handle() {
	return __sync_add_and_fetch(&next_handle, 1);
}
pthread_mutex_t client_tree_lock = PTHREAD_MUTEX_INITIALIZER;
int64_t handle_comparator(void *a, const void *b, void *arg)
{
	uint64_t a_ = *(uint64_t *)a;
	uint64_t b_ = *(uint64_t *)b;
	if (a_ < b_)
		return -1;
	if (a_ > b_)
		return 1;
	return 0;
}
int64_t name_comparator(void *a, const void *b, void *arg)
{
	return strcmp(a, b);
}

int64_t uuid_comparator(void *a, const void *b, void *arg)
{
	return uuid_compare(a, b);
}

static struct ods_rbt client_tree = ODS_RBT_INITIALIZER( handle_comparator );

static bool_t authenticate_request(struct svc_req *rqstp, const char *op)
{
	struct authsys_parms *sys_cred;
	struct authdes_parms *des_cred;

	switch(rqstp->rq_cred.oa_flavor) {
	case AUTH_SYS:
		sys_cred = (struct authsys_parms *) rqstp->rq_clntcred;
		syslog(LOG_INFO, "DSOS AUTH_SYS[%s]: node %s, uid %d, gid %d\n",
		       op,
		       sys_cred->aup_machname,
		       sys_cred->aup_uid,
		       sys_cred->aup_gid);
		break;
	case AUTH_DES:
		des_cred = (struct authdes_parms *) rqstp->rq_clntcred;
		syslog(LOG_ERR, "%s: des_cred %p\n", __func__, des_cred);
		break;
	default:
		svcerr_weakauth(rqstp->rq_xprt);
		return FALSE;
	}
	return TRUE;
}

static void query_destroy(struct dsos_session *client, struct dsos_query *query);

static void session_close(struct dsos_session *client)
{
	/* Remove the client from the client tree */
	pthread_mutex_lock(&client_tree_lock);
	ods_rbt_del(&client_tree, &client->rbn);
	pthread_mutex_unlock(&client_tree_lock);

	/* Clean up any open iterators */
	struct ods_rbn *rbn;
	rbn = ods_rbt_min(&client->iter_tree);
	while (rbn) {
		ods_rbt_del(&client->iter_tree, rbn);
		struct dsos_iter *iter = container_of(rbn, struct dsos_iter, rbn);
		sos_iter_free(iter->iter);
		free(iter);
		rbn = ods_rbt_min(&client->iter_tree);
	}

	/* Free any cached schema */
	rbn = ods_rbt_min(&client->schema_id_tree);
	while (rbn) {
		struct dsos_schema *schema = container_of(rbn, struct dsos_schema, id_rbn);
		ods_rbt_del(&client->schema_id_tree, &schema->id_rbn);
		ods_rbt_del(&client->schema_name_tree, &schema->name_rbn);
		ods_rbt_del(&client->schema_uuid_tree, &schema->uuid_rbn);
	// 	sos_schema_free(schema->schema);
		free(schema);
		rbn = ods_rbt_min(&client->schema_id_tree);
	}

	/* Free up any query */
	rbn = ods_rbt_min(&client->query_tree);
	while (rbn) {
		struct dsos_query *query = container_of(rbn, struct dsos_query, rbn);
		query_destroy(client, query);
		rbn = ods_rbt_min(&client->query_tree);
	}
	/* Close the container */
	sos_container_close(client->sos, SOS_COMMIT_SYNC);

	free(client);
}

static void rderr_handler(SVCXPRT *xprt, void *arg)
{
	struct dsos_session *client = arg;
	syslog(LOG_INFO, "DSOS CLIENT DISCONNECT: xprt %p, client %p\n",
		xprt, client);
	session_close(client);
}

bool_t
open_1_svc(char *path, int perm, int mode, dsos_open_res *res,  struct svc_req *req)
{
	char err_msg[256];
	if (!authenticate_request(req, __func__))
		return FALSE;
	sos_t sos = sos_container_open(path, perm, mode);
	if (!sos) {
		res->error = errno;
		snprintf(err_msg, sizeof(err_msg), "Error %d opening the container", errno);
		res->dsos_open_res_u.error_msg = strdup(err_msg);
		return TRUE;
	}
	struct dsos_session *client = malloc(sizeof *client);
	if (!client) {
		res->error = errno;
		snprintf(err_msg, sizeof(err_msg), "Error %d opening the container", errno);
		res->dsos_open_res_u.error_msg = strdup(err_msg);
		sos_container_close(sos, SOS_COMMIT_ASYNC);
		return TRUE;
	}
	client->handle = get_next_handle();
	client->xprt = req->rq_xprt;
	client->sos = sos;
	client->x_start_time = ZERO_TIME;
	client->x_end_time = ZERO_TIME;
	pthread_mutex_init(&client->schema_tree_lock, NULL);
	ods_rbt_init(&client->schema_id_tree, handle_comparator, NULL);
	ods_rbt_init(&client->schema_uuid_tree, uuid_comparator, NULL);
	ods_rbt_init(&client->schema_name_tree, name_comparator, NULL);
	pthread_mutex_init(&client->iter_tree_lock, NULL);
	ods_rbt_init(&client->iter_tree, handle_comparator, NULL);
	pthread_mutex_init(&client->query_tree_lock, NULL);
	ods_rbt_init(&client->query_tree, handle_comparator, NULL);
	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	client->open_time = client->acc_time;
	ods_rbn_init(&client->rbn, &client->handle);
	pthread_mutex_lock(&client_tree_lock);
	ods_rbt_ins(&client_tree, &client->rbn);
	res->error = 0;
	res->dsos_open_res_u.cont = client->handle;
	pthread_mutex_unlock(&client_tree_lock);
#if 0
	{
		int mode = RPC_SVC_MT_AUTO;
		int max = 20;      /* Set maximum number of threads to 20 */
		if (!rpc_control(RPC_SVC_MTMODE_SET, &mode)) {
			printf("RPC_SVC_MTMODE_SET: failed\n");
			exit(1);
		}
		if (!rpc_control(RPC_SVC_THRMAX_SET, &max)) {
			printf("RPC_SVC_THRMAX_SET: failed\n");
			exit(1);
		}
	}
#endif
	struct svc_rderrhandler rderr;
	rderr.rderr_fn = rderr_handler;
	rderr.rderr_arg = client;
	SVC_CONTROL(req->rq_xprt, SVCSET_RDERRHANDLER, &rderr);
	return TRUE;
}

bool_t close_1_svc(uint64_t handle, int *res, struct svc_req *req)
{
	*res = ENOENT;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	if (!authenticate_request(req, __func__))
		return FALSE;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &handle);
	pthread_mutex_unlock(&client_tree_lock);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	session_close(client);
	/* Remove the client from the client tree */
	*res = 0;
out:
	return TRUE;
}
	/* Clean up any open iterators */

bool_t commit_1_svc(uint64_t handle, int *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	*res = ENOENT;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &handle);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	sos_container_commit(client->sos, SOS_COMMIT_SYNC);
	*res = 0;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t transaction_begin_1_svc(dsos_container_id cont, int *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	*res = ENOENT;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &cont);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	if (client->x_start_time.tv_sec) {
		*res = EBUSY;
		goto out;
	}
	clock_gettime(CLOCK_REALTIME, &client->x_start_time);
	client->x_end_time = ZERO_TIME;
	*res = 0;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t transaction_end_1_svc(dsos_container_id cont, int *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	*res = ENOENT;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &cont);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	if (client->x_end_time.tv_sec) {
		*res = EINVAL;
		goto out;
	}
	client->x_start_time = ZERO_TIME;
	sos_container_commit(client->sos, SOS_COMMIT_SYNC);
	clock_gettime(CLOCK_REALTIME, &client->x_end_time);
	*res = 0;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}
static struct dsos_session *get_client(dsos_container_id handle)
{
	struct ods_rbn *rbn;
	struct dsos_session *client = NULL;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &handle);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
out:
	pthread_mutex_unlock(&client_tree_lock);
	return client;
}
static void put_client(struct dsos_session *client) {}

static struct dsos_schema *get_schema(struct dsos_session *client, dsos_schema_id schema_handle)
{
	struct ods_rbn *schema_rbn;
	struct dsos_schema *schema = NULL;

	pthread_mutex_lock(&client->schema_tree_lock);
	schema_rbn = ods_rbt_find(&client->schema_id_tree, &schema_handle);
	if (schema_rbn)
		schema = container_of(schema_rbn, struct dsos_schema, id_rbn);
	pthread_mutex_unlock(&client->schema_tree_lock);
	return schema;
}
static void put_schema(struct dsos_schema *schema) {}

static struct dsos_iter *get_iter(struct dsos_session *client, dsos_iter_id iter_handle)
{
	struct dsos_iter *iter = NULL;
	struct ods_rbn *iter_rbn;
	pthread_mutex_lock(&client->iter_tree_lock);
	iter_rbn = ods_rbt_find(&client->iter_tree, &iter_handle);
	if (iter_rbn)
		iter = container_of(iter_rbn, struct dsos_iter, rbn);
	pthread_mutex_unlock(&client->iter_tree_lock);
	return iter;
}
static void put_iter(struct dsos_iter *iter) {}

static struct dsos_query *get_query(struct dsos_session *client, dsos_query_id query_id)
{
	struct dsos_query *query = NULL;
	struct ods_rbn *query_rbn;
	pthread_mutex_lock(&client->query_tree_lock);
	query_rbn = ods_rbt_find(&client->query_tree, &query_id);
	if (query_rbn)
		query = container_of(query_rbn, struct dsos_query, rbn);
	pthread_mutex_unlock(&client->query_tree_lock);
	return query;
}
static void put_query(struct dsos_query *query) {}

static struct dsos_schema *
cache_schema(struct dsos_session *client, sos_schema_t schema)
{
	struct dsos_schema *dschema;
	/* See if it's already cached */
	pthread_mutex_lock(&client->schema_tree_lock);
	struct ods_rbn *rbn = ods_rbt_find(&client->schema_name_tree, sos_schema_name(schema));
	pthread_mutex_unlock(&client->schema_tree_lock);
	if (rbn)
		return container_of(rbn, struct dsos_schema, name_rbn);
	dschema = malloc(sizeof *dschema);
	if (!dschema)
		return NULL;
	dschema->spec = dsos_spec_from_schema(schema);
	if (!dschema->spec)
		goto err_0;
	dschema->spec->id = dschema->handle = get_next_handle();
	dschema->schema = schema;
	pthread_mutex_lock(&client->schema_tree_lock);
	/* Id tree */
	ods_rbn_init(&dschema->id_rbn, &dschema->handle);
	ods_rbt_ins(&client->schema_id_tree, &dschema->id_rbn);
	/* Name tree */
	ods_rbn_init(&dschema->name_rbn, dschema->spec->name);
	ods_rbt_ins(&client->schema_name_tree, &dschema->name_rbn);
	/* UUID tree */
	ods_rbn_init(&dschema->uuid_rbn, dschema->spec->uuid);
	ods_rbt_ins(&client->schema_uuid_tree, &dschema->uuid_rbn);
	pthread_mutex_unlock(&client->schema_tree_lock);

	return dschema;
err_0:
	free(dschema);
	return NULL;
}
bool_t schema_create_1_svc(dsos_container_id cont_id, dsos_schema_spec spec, dsos_schema_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	sos_schema_t schema;
	int rc;
	struct dsos_schema *dschema;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}
	schema = dsos_schema_from_spec(&spec);
	if (!schema) {
		res->error = errno;
		goto out_1;
	}
	rc = sos_schema_add(client->sos, schema);
	if (rc) {
		res->error = rc;
		goto out_2;
	}
	dschema = cache_schema(client, schema);
	if (!dschema) {
		res->error = errno;
		goto out_2;
	}
	spec.id = dschema->spec->id;
	res->error = 0;
	res->dsos_schema_res_u.spec = dsos_schema_spec_dup(&spec);
	assert(res->dsos_schema_res_u.spec);
	return TRUE;
out_2:
	sos_schema_free(schema);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t schema_find_by_id_1_svc(dsos_container_id cont_id, dsos_schema_id schema_id, dsos_schema_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	rbn = ods_rbt_find(&client->schema_id_tree, &schema_id);
	if (!rbn) {
		res->error = ENOENT;
		goto out_1;
	}
	dschema = container_of(rbn, struct dsos_schema, id_rbn);
	res->error = 0;
	res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t schema_find_by_name_1_svc(dsos_container_id cont_id, char *name, dsos_schema_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;
	sos_schema_t schema;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	rbn = ods_rbt_find(&client->schema_name_tree, name);
	if (rbn) {
		res->error = 0;
		dschema = container_of(rbn, struct dsos_schema, name_rbn);
		res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
		goto out_1;
	}
	schema = sos_schema_by_name(client->sos, name);
	if (!schema) {
		res->error = errno;
		goto  out_1;
	}
	dschema = cache_schema(client, schema);
	if (!dschema) {
		res->error = errno;
		goto out_1;
	}
	res->error = 0;
	res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t schema_find_by_uuid_1_svc(dsos_container_id cont_id, char *uuid, dsos_schema_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;
	sos_schema_t schema;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	rbn = ods_rbt_find(&client->schema_uuid_tree, uuid);
	if (rbn) {
		res->error = 0;
		dschema = container_of(rbn, struct dsos_schema, uuid_rbn);
		res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
		goto out_1;
	}
	schema = sos_schema_by_uuid(client->sos, (unsigned char *)uuid);
	if (!schema) {
		res->error = errno;
		goto  out_1;
	}
	dschema = cache_schema(client, schema);
	if (!dschema) {
		res->error = errno;
		goto out_1;
	}
	res->error = 0;
	res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t schema_query_1_svc(dsos_container_id cont_id, dsos_schema_query_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	struct dsos_session *client;
	int count, array_size;
	sos_schema_t schema;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	array_size = 0;
	for (schema = sos_schema_first(client->sos); schema; schema = sos_schema_next(schema))
		array_size += 1;

	res->dsos_schema_query_res_u.names.names_val = calloc(array_size, sizeof(char *));
	res->dsos_schema_query_res_u.names.names_len = array_size;
	res->error = 0;
	count = 0;
	for (schema = sos_schema_first(client->sos); schema; schema = sos_schema_next(schema)) {
		res->dsos_schema_query_res_u.names.names_val[count] = strdup(sos_schema_name(schema));
		count += 1;
	}
out_0:
	return TRUE;
}

bool_t obj_create_1_svc(dsos_obj_link obj_list, dsos_create_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_schema *schema;
	dsos_obj_entry *obj_e;

	obj_e = obj_list;
	while (obj_e) {
		client = get_client(obj_e->cont_id);
		if (!client) {
			res->error = DSOS_ERR_CLIENT;
			goto out_0;
		}
		schema = get_schema(client, obj_e->schema_id);
		if (!schema) {
			res->error = DSOS_ERR_SCHEMA;
			goto out_1;
		}
		clock_gettime(CLOCK_REALTIME, &client->acc_time);
		sos_obj_t obj = sos_obj_new_with_data(
						schema->schema,
						obj_e->value.dsos_obj_value_val,
						obj_e->value.dsos_obj_value_len);
		if (!obj) {
			res->error = errno;
			goto out_2;
		}
		sos_obj_commit(obj);
		sos_obj_index(obj);
		sos_obj_put(obj);
		obj_e = obj_e->next;
		res->error = 0;
		res->dsos_create_res_u.obj_id = 0; // object->handle;
	}
out_2:
	put_schema(schema);
out_1:
	put_client(client);
out_0:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t obj_delete_1_svc(dsos_container_id cont, dsos_obj_id obj, int *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	*res = ENOTSUP;
	return TRUE;
}

bool_t iter_create_1_svc(dsos_container_id cont_id, dsos_schema_id schema_id, dsos_attr_name attr_name,
			dsos_iter_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	sos_iter_t iter;
	struct dsos_iter *diter;
	struct dsos_session *client;
	struct dsos_schema *schema;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	schema = get_schema(client, schema_id);
	if (!schema) {
		res->error = DSOS_ERR_SCHEMA;
		goto out_1;
	}

	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	sos_attr_t attr = sos_schema_attr_by_name(schema->schema, attr_name);
	if (!attr) {
		res->error = DSOS_ERR_ATTR;
		goto out_2;
	}

	iter = sos_attr_iter_new(attr);
	if (!iter) {
		res->error = errno;
		goto out_2;
	}

	diter = malloc(sizeof *diter);
	if (!diter) {
		res->error = errno;
		goto out_3;
	}

	diter->cont_id = cont_id;
	diter->schema_id = schema_id;
	diter->iter = iter;
	diter->handle = get_next_handle();
	ods_rbn_init(&diter->rbn, &diter->handle);
	ods_rbt_ins(&client->iter_tree, &diter->rbn);
	res->error = 0;
	res->dsos_iter_res_u.iter_id = diter->handle;
	goto out_2;

out_3:
	sos_iter_free(iter);
out_2:
	put_schema(schema);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_delete_1_svc(dsos_container_id cont, dsos_iter_id iter_id, int *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_iter *diter;

	client = get_client(cont);
	if (!client) {
		*res = DSOS_ERR_CLIENT;
		goto out_0;
	}

	diter = get_iter(client, iter_id);
	if (!diter) {
		*res = DSOS_ERR_ITER;
		goto out_1;
	}

	pthread_mutex_lock(&client->iter_tree_lock);
	ods_rbt_del(&client->iter_tree, &diter->rbn);
	pthread_mutex_unlock(&client->iter_tree_lock);
	put_iter(diter);
	*res = 0;
out_1:
	put_client(client);
out_0:
	return TRUE;
}

static int __make_obj_list(dsos_obj_list_res *result, struct dsos_iter *diter)
{
	int count = 5;
	struct dsos_obj_entry *entry = NULL;
	result->error = DSOS_ERR_ITER_EMPTY;
	int rc = 0;
	while (!rc && count) {
		result->error = 0;
		sos_obj_t obj = sos_iter_obj(diter->iter);
		if (!obj) {
			result->error = errno;
			goto err_0;
		}
		if (entry) {
			entry->next = malloc(sizeof *entry);
			entry = entry->next;
		} else {
			entry = malloc(sizeof *entry);
			result->dsos_obj_list_res_u.obj_list = entry;
		}
		entry->next = NULL;
		entry->cont_id = diter->cont_id;
		entry->schema_id = diter->schema_id;
		count --;
		void *obj_data = sos_obj_ptr(obj);
		entry->value.dsos_obj_value_len = sos_obj_size(obj);
		entry->value.dsos_obj_value_val = malloc(entry->value.dsos_obj_value_len);
		memcpy(entry->value.dsos_obj_value_val, obj_data, entry->value.dsos_obj_value_len);
		sos_obj_put(obj);
		if (count)
			rc = sos_iter_next(diter->iter);
	}
	return 0;
err_0:
	return -1;
}

bool_t iter_begin_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_iter *diter;
	int rc;

	client = get_client(cont);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	diter = get_iter(client, iter_id);
	if (!diter) {
		res->error = DSOS_ERR_ITER;
		goto out_1;
	}

	clock_gettime(CLOCK_REALTIME, &client->acc_time);

	rc = sos_iter_begin(diter->iter);
	if (!rc) {
		rc = __make_obj_list(res, diter);
	} else {
		res->error = rc;
	}
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_end_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}

bool_t iter_next_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_iter *diter;
	int rc;

	client = get_client(cont);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	diter = get_iter(client, iter_id);
	if (!diter) {
		res->error = DSOS_ERR_ITER;
		goto out_1;
	}

	clock_gettime(CLOCK_REALTIME, &client->acc_time);

	rc = sos_iter_next(diter->iter);
	if (!rc) {
		rc = __make_obj_list(res, diter);
	} else {
		res->error = rc;
	}
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_prev_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_glb_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_lub_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}

bool_t query_create_1_svc(dsos_container_id cont_id, dsos_query_options opts, dsos_query_create_res *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_query *query;
	struct dsos_session *client;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	query = calloc(1, sizeof *query);
	if (!query) {
		res->error = errno;
		goto out_1;
	}
	query->state = DSOSQ_STATE_INIT;
	query->cont_id = cont_id;
	query->handle = get_next_handle();
	query->ast = ast_create(client->sos, query->handle);
	ods_rbn_init(&query->rbn, &query->handle);
	ods_rbt_ins(&client->query_tree, &query->rbn);
	res->error = 0;
	res->dsos_query_create_res_u.query_id = query->handle;

out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t query_select_1_svc(dsos_container_id cont_id, dsos_query_id query_id, dsos_query query_str, dsos_query_select_res *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_query *query;
	int rc;
	char err_msg[256];
	client = get_client(cont_id);
	memset(res, 0, sizeof(*res));
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		sprintf(err_msg, "Invalid container id %ld\n", cont_id);
		res->dsos_query_select_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	query = get_query(client, query_id);
	if (!query) {
		res->error = DSOS_ERR_QUERY_ID;
		sprintf(err_msg, "Invalid query id %ld\n", query_id);
		res->dsos_query_select_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	rc = ast_parse(query->ast, query_str);
	res->error = rc;
	if (rc) {
		res->dsos_query_select_res_u.error_msg = strdup(query->ast->error_msg);
	} else {
		sos_attr_t res_key_attr =
			sos_schema_attr_by_name(query->ast->result_schema,
						sos_attr_name(query->
							      ast->iter_attr_e->sos_attr));
		res->dsos_query_select_res_u.select.key_attr_id = sos_attr_id(res_key_attr);
		res->dsos_query_select_res_u.select.spec = dsos_spec_from_schema(query->ast->result_schema);
	}
	if (0 == res->error)
		query->state = DSOSQ_STATE_BEGIN;
	put_query(query);
out_0:
	return TRUE;
}

#define QUERY_OBJECT_COUNT	4096
static int __make_query_obj_list(struct dsos_session *client, struct ast *ast,
				dsos_query_next_res *result)
{
	enum ast_eval_e eval;
	sos_iter_t iter = ast->sos_iter;
	int count = QUERY_OBJECT_COUNT;
	struct dsos_obj_entry *entry = NULL;
	result->error = DSOS_ERR_QUERY_EMPTY;
	int rc = 0;
	memset(result, 0, sizeof(*result));
	ast_attr_entry_t attr_e;

	while (!rc && count) {
		sos_obj_t obj = sos_iter_obj(iter);
		if (!obj) {
			result->error = errno;
			goto err_0;
		}
		eval = ast_eval(ast, obj);
		switch (eval) {
		case AST_EVAL_NOMATCH:
			sos_obj_put(obj);
			rc = sos_iter_next(iter);
			continue;
		case AST_EVAL_EMPTY:
			sos_obj_put(obj);
			goto out;
		case AST_EVAL_MATCH:
			break;
		}
		sos_obj_t result_obj = sos_obj_malloc(ast->result_schema);
		TAILQ_FOREACH(attr_e, &ast->select_list, link) {
			sos_obj_attr_copy(result_obj, attr_e->res_attr, obj, attr_e->sos_attr);
		}
		result->error = 0;
		if (entry) {
			entry->next = malloc(sizeof *entry);
			entry = entry->next;
		} else {
			entry = malloc(sizeof *entry);
			result->dsos_query_next_res_u.result.obj_list = entry;
		}
		result->dsos_query_next_res_u.result.count += 1;
		result->dsos_query_next_res_u.result.format = 0;
		entry->next = NULL;
		entry->cont_id = client->handle;
		entry->schema_id = 0;
		count --;
		void *obj_data = sos_obj_ptr(result_obj);
		entry->value.dsos_obj_value_len = sos_obj_size(result_obj);
		entry->value.dsos_obj_value_val = malloc(entry->value.dsos_obj_value_len);
		memcpy(entry->value.dsos_obj_value_val, obj_data, entry->value.dsos_obj_value_len);
		sos_obj_put(obj);
		sos_obj_put(result_obj);
		if (count)
			rc = sos_iter_next(iter);
	}
out:
	return 0;
err_0:
	return -1;
}

struct ast_term *query_find_term(struct dsos_query *query, sos_attr_t filt_attr)
{
	return ast_find_term(query->ast->where, sos_attr_name(filt_attr));
}

bool_t query_next_1_svc(dsos_container_id cont_id, dsos_query_id query_id, dsos_query_next_res *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_query *query;
	int rc;
	char err_msg[256];
	SOS_KEY(key);

	client = get_client(cont_id);
	if (!client) {
		sprintf(err_msg, "Invalid container id %ld", cont_id);
		res->error = DSOS_ERR_CLIENT;
		res->dsos_query_next_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	clock_gettime(CLOCK_REALTIME, &client->acc_time);
	query = get_query(client, query_id);
	if (!query) {
		sprintf(err_msg, "Invalid query id %ld", query_id);
		res->error = DSOS_ERR_QUERY_ID;
		res->dsos_query_next_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	switch (query->state) {
	case DSOSQ_STATE_INIT:
		sprintf(err_msg,
			"There is no valid 'select' pending on query %ld",
			query_id);
		res->error = DSOS_ERR_QUERY_BAD_SELECT;
		res->dsos_query_next_res_u.error_msg = strdup(err_msg);
		break;
	case DSOSQ_STATE_BEGIN:
		rc = ast_start_key(query->ast, key);
		if (rc == ESRCH) {
			rc = sos_iter_sup(query->ast->sos_iter, key);
		} else {
			rc = sos_iter_begin(query->ast->sos_iter);
		}
		if (rc)
			goto empty;
		rc = __make_query_obj_list(client, query->ast, res);
		if (!rc)
			query->state = DSOSQ_STATE_NEXT;
		else
			goto empty;
		break;
	case DSOSQ_STATE_NEXT:
		rc = sos_iter_next(query->ast->sos_iter);
		if (rc)
			goto empty;
		rc = __make_query_obj_list(client, query->ast, res);
		if (rc)
			goto empty;
		break;
	case DSOSQ_STATE_EMPTY:
		goto empty;
	}
	put_query(query);
out_0:
	return TRUE;
empty:
	sprintf(err_msg, "No more data for query %ld.", query_id);
	res->error = DSOS_ERR_QUERY_EMPTY;
	res->dsos_query_next_res_u.error_msg = strdup(err_msg);
	put_query(query);
	return TRUE;
}

static void query_destroy(struct dsos_session *client, struct dsos_query *query)
{

 	pthread_mutex_lock(&client->query_tree_lock);
	ods_rbt_del(&client->query_tree, &query->rbn);
	pthread_mutex_unlock(&client->query_tree_lock);

	ast_destroy(query->ast);
	free(query);
}

bool_t query_destroy_1_svc(dsos_container_id cont_id, dsos_query_id query_id,
			   dsos_query_destroy_res *res, struct svc_req *rqst)
{
	struct dsos_session *client;
	struct dsos_query *query;
	int rc;
	char err_msg[256];

	memset(res, 0, sizeof(*res));
	if (!authenticate_request(rqst, __func__))
		return FALSE;

	client = get_client(cont_id);
	if (!client) {
		sprintf(err_msg, "Invalid container id %ld", cont_id);
		res->error = DSOS_ERR_CLIENT;
		res->dsos_query_destroy_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	query = get_query(client, query_id);
	if (!query) {
		sprintf(err_msg, "Invalid query id %ld", query_id);
		res->error = DSOS_ERR_QUERY_ID;
		res->dsos_query_destroy_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}

	query_destroy(client, query);
 out_0:
	return TRUE;
}

int sosdb_1_freeresult (SVCXPRT *transp, xdrproc_t xdr_result, caddr_t result)

{
	xdr_free (xdr_result, result);

	/*
	 * Insert additional freeing code here, if needed
	 */

	return 1;
}
