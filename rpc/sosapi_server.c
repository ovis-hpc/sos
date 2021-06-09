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
#include "sosapi.h"          /* Created by rpcgen */
#include "sosapi_common.h"

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
	struct ods_rbn rbn;
};

struct dsos_object {
	uint64_t handle;
	sos_obj_t obj;
	struct ods_rbn rbn;
};

struct dsos_iter {
	uint64_t handle;
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

static uint64_t next_handle = 1;
static inline get_next_handle() {
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

static void session_close(struct dsos_session *client)
{
	struct ods_rbn *rbn;
	rbn = ods_rbt_min(&client->iter_tree);
	while (rbn) {
		ods_rbt_del(&client->iter_tree, rbn);
		struct dsos_iter *iter = container_of(rbn, struct dsos_iter, rbn);
		sos_iter_free(iter->iter);
		free(iter);
		rbn = ods_rbt_min(&client->iter_tree);
	}

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

	// TODO: Object Tree -- not sure we'll do this

	/* Close the container */
	sos_container_close(client->sos, SOS_COMMIT_SYNC);
	ods_rbt_del(&client_tree, &client->rbn);
	free(client);
}

void idle_client_cleanup(const SVCXPRT *handle, const bool_t b)
{
	struct ods_rbn *rbn = ods_rbt_min(&client_tree);
	while (rbn) {
		struct ods_rbn *next = ods_rbn_succ(rbn);
		struct dsos_session *client = container_of(rbn, struct dsos_session, rbn);
		if (client->xprt == handle) {
			session_close(client);
			ods_rbt_del(&client_tree, rbn);
		}
		rbn = next;
	}
}

bool_t
open_1_svc(char *path, int perm, int mode, dsos_open_res *res,  struct svc_req *req)
{
	sos_t sos = sos_container_open(path, perm, mode);
	if (!sos) {
		res->error = errno;
		return TRUE;
	}
	struct dsos_session *client = malloc(sizeof *client);
	if (!client) {
		sos_container_close(sos, SOS_COMMIT_ASYNC);
		res->error = ENOMEM;
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
	return TRUE;
}

bool_t close_1_svc(uint64_t handle, int *res, struct svc_req *req)
{
	*res = ENOENT;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &handle);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	session_close(client);
	*res = 0;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t commit_1_svc(uint64_t handle, int *res, struct svc_req *req)
{
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
static put_client(struct dsos_session *client) {}

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

#if 0
static struct dsos_object *get_object(struct dsos_session *client, dsos_obj_id obj_handle)
{
	struct ods_rbn *obj_rbn = ods_rbt_find(&client->obj_tree, &obj_handle);
	if (!obj_rbn)
		return NULL;
	return container_of(obj_rbn, struct dsos_object, rbn);
}
static void put_object(struct dsos_object *object) {}
#endif

static struct dsos_schema *
cache_schema(struct dsos_session *client, sos_schema_t schema)
{
	int rc;
	struct dsos_schema *dschema = malloc(sizeof *dschema);
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
bool_t schema_create_1_svc(dsos_container_id cont, dsos_schema_spec spec, dsos_schema_res *res, struct svc_req *req)
{
	struct dsos_session *client;
	sos_schema_t schema;
	int rc;
	struct dsos_schema *dschema;

	client = get_client(cont);
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

bool_t schema_find_by_id_1_svc(dsos_container_id cont, dsos_schema_id schema_id, dsos_schema_res *res, struct svc_req *rqstp)
{
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;
	int rc;

	client = get_client(cont);
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

bool_t schema_find_by_name_1_svc(dsos_container_id cont, char *name, dsos_schema_res *res, struct svc_req *rqstp)
{
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;
	sos_schema_t schema;
	int rc;

	client = get_client(cont);
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
		goto out_1;
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

bool_t schema_find_by_uuid_1_svc(dsos_container_id cont, char *uuid, dsos_schema_res *res, struct svc_req *rqstp)
{
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_schema *dschema;
	sos_schema_t schema;
	int rc;

	client = get_client(cont);
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
	schema = sos_schema_by_uuid(client->sos, uuid);
	if (!schema) {
		res->error = errno;
		goto out_1;
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

bool_t obj_create_1_svc(dsos_obj_link obj_list, dsos_create_res *res, struct svc_req *req)
{
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
		res->dsos_create_res_u.obj = 0; // object->handle;
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
	*res = ENOTSUP;
	return TRUE;
}

bool_t iter_create_1_svc(dsos_container_id cont, dsos_schema_id schema_id, dsos_attr_name attr_name,
	dsos_iter_res *res, struct svc_req *rqstp)
{
	sos_iter_t iter;
	struct dsos_iter *diter;
	struct dsos_session *client;
	struct dsos_schema *schema;

	client = get_client(cont);
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

	diter->iter = iter;
	diter->handle = get_next_handle();
	ods_rbn_init(&diter->rbn, &diter->handle);
	ods_rbt_ins(&client->iter_tree, &diter->rbn);
	res->error = 0;
	res->dsos_iter_res_u.iter = diter->handle;
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
	*res = 0;
out_1:
	put_client(client);
out_0:
	return TRUE;
}

int __make_obj_list(dsos_obj_list_res *result, struct dsos_iter *diter)
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
	struct dsos_session *client;
	struct dsos_iter *diter;
	int rc, count;

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
	return TRUE;
}

bool_t iter_next_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_list_res *res, struct svc_req *req)
{
	struct dsos_session *client;
	struct dsos_iter *diter;
	int rc, count;

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
	return TRUE;
}
bool_t iter_find_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	return TRUE;
}
bool_t iter_find_glb_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
	return TRUE;
}
bool_t iter_find_lub_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_list_res *res, struct svc_req *req)
{
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
