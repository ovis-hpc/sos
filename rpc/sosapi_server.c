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
#include <unistd.h>
#include <uuid/uuid.h>
#include <jansson.h>
#include "dsos.h"
#include "sosapi.h"          /* Created by rpcgen */
#include "sosapi_common.h"
#include "ast.h"

static struct timespec ZERO_TIME = {
	0, 0
};

static int dir_read = 0;
struct dir_entry_s {
	char *name;
	char *path;
	struct ods_rbn rbn;
};

static int64_t name_comparator(void *a, const void *b, void *arg);
static struct ods_rbt dir_tree = ODS_RBT_INITIALIZER(name_comparator);

/*
 * Map a container name to a local path
 *
 * If a name mapping exists, return the mapped name, otherwise
 * return the name argument.
 */
static char *get_container_path(char *name)
{
	struct ods_rbn *rbn = ods_rbt_find(&dir_tree, name);
	if (!rbn)
		return name;

	struct dir_entry_s *entry = container_of(rbn, struct dir_entry_s, rbn);
	return entry->path;
}

static void read_directory(void)
{
	const char *path = getenv("DSOSD_DIRECTORY");
	const char *server_id = getenv("DSOSD_SERVER_ID");
	char hostname[PATH_MAX];
	json_t *dir, *entry;
	json_t *server;
	json_error_t error;
	struct dir_entry_s *dir_entry;
	int rc;
	FILE *dir_file;
	if (dir_read)
		return;
	dir_read = 1;
	if (!path)
		return;
	dir_file = fopen(path, "r");
	if (!dir_file)
		return;
	dir = json_loadf(dir_file, 0, &error);
	if (!dir) {
		printf("Error parsing container directory:\n");
		printf("    %s\n", error.text);
		printf("    %s\n", error.source);
		printf("    %*s\n", error.column, "^");
		return;
	}
	/* Look up the container name mappings for this hostname */
	if (!server_id) {
		rc = gethostname(hostname, PATH_MAX);
		if (rc)
			return;
		server_id = hostname;
	}
	if (!json_is_object(dir)) {
		printf("Expected a dictionary, the directory is invalid.\n");
		return;
	}
	server = json_object_get(dir, server_id);
	if (!server)
		return;
	if (!json_is_object(server)) {
		printf("Expected a directory with name %s but the entry value is a %d\n",
		       server_id, json_typeof(server));
		return;
	}
	const char *name;
	json_object_foreach(server, name, entry) {
		if (!json_is_string(entry)) {
			printf("Expected \"<container-name>\" : \"<container-path>\", "
			       "but the entry is not a string\n");
			continue;
		}
		dir_entry = malloc(sizeof(*dir_entry));
		if (!dir_entry) {
			printf("%s[%d] : Memory allocation failure\n", __func__, __LINE__);
			continue;
		}
		dir_entry->name = strdup(name);
		if (!dir_entry->name)
			assert(0 == "Memory allocation failure.");
		dir_entry->path = strdup(json_string_value(entry));
		if (!dir_entry->path)
			assert(0 == "Memory allocation failure.");
		ods_rbn_init(&dir_entry->rbn, dir_entry->name);
		ods_rbt_ins(&dir_tree, &dir_entry->rbn);
	}
}

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
	pthread_mutex_t part_tree_lock;
	struct ods_rbt part_id_tree;
	struct ods_rbt part_name_tree;
	struct ods_rbt part_uuid_tree;
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
	uint32_t count;		/* number of objects to return at a time */
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

struct dsos_part {
	uint64_t handle;
	sos_part_t part;
	struct ods_rbn id_rbn;
	struct ods_rbn name_rbn;
	struct ods_rbn uuid_rbn;
	dsos_part_spec spec;
};

struct dsos_query {
	uint64_t handle;
	enum dsos_query_state_e {
		DSOSQ_STATE_INIT = 0,
		DSOSQ_STATE_BEGIN,
		DSOSQ_STATE_NEXT,
		DSOSQ_STATE_EMPTY
	} state;
	uint32_t count;		/* number of objects to return at a time */
	dsos_container_id cont_id;
	struct ast *ast;
	struct ods_rbn rbn;
};

static uint64_t next_handle = 1;
static inline uint64_t get_next_handle() {
	uint64_t maybe = __sync_add_and_fetch(&next_handle, 1);
	return maybe ? maybe : __sync_add_and_fetch(&next_handle, 1);
}
pthread_mutex_t client_tree_lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t handle_comparator(void *a, const void *b, void *arg)
{
	uint64_t a_ = *(uint64_t *)a;
	uint64_t b_ = *(uint64_t *)b;
	if (a_ < b_)
		return -1;
	if (a_ > b_)
		return 1;
	return 0;
}
static int64_t name_comparator(void *a, const void *b, void *arg)
{
	return strcmp(a, b);
}

static int64_t uuid_comparator(void *a, const void *b, void *arg)
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
		assert(0 == "unsupported authentication");
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

	/* Free any cached partitions */
	rbn = ods_rbt_min(&client->part_id_tree);
	while (rbn) {
		struct dsos_part *part = container_of(rbn, struct dsos_part, id_rbn);
		ods_rbt_del(&client->part_id_tree, &part->id_rbn);
		ods_rbt_del(&client->part_name_tree, &part->name_rbn);
		ods_rbt_del(&client->part_uuid_tree, &part->uuid_rbn);
		sos_part_put(part->part);
		rbn = ods_rbt_min(&client->part_id_tree);
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
	struct authsys_parms *sys_cred;
	uid_t euid;
	gid_t egid;
	char err_msg[256];
	sos_t sos;
	char *lpath;
	struct svc_rderrhandler rderr;

	if (!authenticate_request(req, __func__))
		return FALSE;

	read_directory();

	/* Map the specified path to the local path */
	lpath = get_container_path(path);
	if (lpath)
		path = lpath;

	switch(req->rq_cred.oa_flavor) {
	case AUTH_SYS:
		sys_cred = (struct authsys_parms *) req->rq_clntcred;
		euid = sys_cred->aup_uid;
		egid = sys_cred->aup_gid;
		syslog(LOG_INFO, "DSOS OPEN(path %s, uid %d, gid %d)\n",
		       path, euid, egid);
		break;
	default:
		svcerr_weakauth(req->rq_xprt);
		return FALSE;
	}
	if (euid != 0 || egid != 0) {
 		/* Create/open the container as a particular user.group */
		perm |= SOS_PERM_USER;
		if (perm & SOS_PERM_CREAT) {
			sos = sos_container_open(path, perm, mode, euid, egid);
		} else {
			sos = sos_container_open(path, perm, euid, egid);
		}
	} else {
		sos = sos_container_open(path, perm, mode);
	}
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
	pthread_mutex_init(&client->part_tree_lock, NULL);
	ods_rbt_init(&client->part_id_tree, handle_comparator, NULL);
	ods_rbt_init(&client->part_uuid_tree, uuid_comparator, NULL);
	ods_rbt_init(&client->part_name_tree, name_comparator, NULL);
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
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	session_close(client);
	/* Remove the client from the client tree */
	ods_rbt_del(&client_tree, rbn);
	*res = 0;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t destroy_1_svc(char *path, int *res, struct svc_req *req)
{
	struct authsys_parms *sys_cred;
	uid_t euid;
	gid_t egid;
	char err_msg[256];
	char *lpath;

	if (!authenticate_request(req, __func__))
		return FALSE;

	read_directory();

	/* Map the specified path to the local path */
	lpath = get_container_path(path);
	if (lpath)
		path = lpath;

	switch(req->rq_cred.oa_flavor) {
	case AUTH_SYS:
		sys_cred = (struct authsys_parms *) req->rq_clntcred;
		euid = sys_cred->aup_uid;
		egid = sys_cred->aup_gid;
		syslog(LOG_INFO, "DSOS DESTROY(path %s, uid %d, gid %d)\n",
		       path, euid, egid);
		break;
	default:
		svcerr_weakauth(req->rq_xprt);
		return FALSE;
	}

	/* TODO: Remove all files associated with the local instance of the container. */
	*res = 0;
	return TRUE;
}

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

bool_t transaction_begin_1_svc(dsos_container_id cont, dsos_timespec tv, dsos_transaction_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	res->error = 0;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &cont);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	if (client->x_start_time.tv_sec) {
		res->error = EBUSY;
		res->dsos_transaction_res_u.error_msg = NULL;
		goto out;
	}
	clock_gettime(CLOCK_REALTIME, &client->x_start_time);
	client->x_end_time = ZERO_TIME;
out:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t transaction_end_1_svc(dsos_container_id cont, dsos_transaction_res *res, struct svc_req *rqstp)
{
	if (!authenticate_request(rqstp, __func__))
		return FALSE;
	res->error = 0;
	struct ods_rbn *rbn;
	struct dsos_session *client;
	pthread_mutex_lock(&client_tree_lock);
	rbn = ods_rbt_find(&client_tree, &cont);
	if (!rbn)
		goto out;
	client = container_of(rbn, struct dsos_session, rbn);
	if (client->x_end_time.tv_sec) {
		res->error = EINVAL;
		res->dsos_transaction_res_u.error_msg = NULL;
		goto out;
	}
	client->x_start_time = ZERO_TIME;
	sos_container_commit(client->sos, SOS_COMMIT_SYNC);
	clock_gettime(CLOCK_REALTIME, &client->x_end_time);
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

static struct dsos_part *get_part(struct dsos_session *client, dsos_part_id part_handle)
{
	struct ods_rbn *part_rbn;
	struct dsos_part *part = NULL;

	pthread_mutex_lock(&client->part_tree_lock);
	part_rbn = ods_rbt_find(&client->part_id_tree, &part_handle);
	if (part_rbn)
		part = container_of(part_rbn, struct dsos_part, id_rbn);
	pthread_mutex_unlock(&client->part_tree_lock);
	return part;
}
static void put_part(struct dsos_part *part) {}

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
	char err_msg[256];

	memset(res, 0, sizeof *res);
	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		sprintf(err_msg, "Invalid container id %d in request.", (int)cont_id);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	schema = dsos_schema_from_spec(&spec);
	if (!schema) {
		res->error = errno;
		sprintf(err_msg, "Error %d decoding schema specification.", res->error);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	rc = sos_schema_add(client->sos, schema);
	if (rc) {
		res->error = rc;
		if (rc == EEXIST)
			sprintf(err_msg, "A schema named '%s' already exists in the container.",
				sos_schema_name(schema));
		else
			sprintf(err_msg, "Error %d adding the schema to the container.", rc);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	dschema = cache_schema(client, schema);
	if (!dschema) {
		res->error = errno;
		sprintf(err_msg, "Error %d caching the schema.", res->error);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
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

bool_t schema_find_by_name_1_svc(dsos_container_id cont_id, char *name,
				dsos_schema_res *res, struct svc_req *rqstp)
{
	char err_msg[256];
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
		sprintf(err_msg, "The container %ld does not exist.", cont_id);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}

	rbn = ods_rbt_find(&client->schema_name_tree, name);
	if (rbn) {
		res->error = 0;
		dschema = container_of(rbn, struct dsos_schema, name_rbn);
		res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
		if (!res->dsos_schema_res_u.spec) {
			res->error = errno ? errno : EINVAL;
			snprintf(err_msg, sizeof(err_msg),
				"Error %d duplicating the schema", res->error);
			res->dsos_schema_res_u.error_msg = strdup(err_msg);
		}
		goto out_1;
	}
	schema = sos_schema_by_name(client->sos, name);
	if (!schema) {
		res->error = errno;
		snprintf(err_msg, sizeof(err_msg),
			"The schema '%s' does not exist in container %ld", name, cont_id);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto  out_1;
	}
	dschema = cache_schema(client, schema);
	if (!dschema) {
		res->error = errno ? errno : EINVAL;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d encoding the schema", res->error);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
		goto  out_1;
	}
	res->error = 0;
	res->dsos_schema_res_u.spec = dsos_schema_spec_dup(dschema->spec);
	if (!res->dsos_schema_res_u.spec) {
		res->error = errno ? errno : EINVAL;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d duplicating the schema", res->error);
		res->dsos_schema_res_u.error_msg = strdup(err_msg);
	}
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

static struct dsos_part *
cache_part(struct dsos_session *client, sos_part_t part)
{
	struct dsos_part *dpart;
	const char *name = sos_part_name(part);
	uuid_t uuid;
	sos_part_uuid(part, uuid);
	/* See if it's already cached */
	pthread_mutex_lock(&client->part_tree_lock);
	struct ods_rbn *rbn = ods_rbt_find(&client->part_name_tree, name);
	pthread_mutex_unlock(&client->part_tree_lock);
	if (rbn)
 		return container_of(rbn, struct dsos_part, name_rbn);
	dpart = malloc(sizeof *dpart);
	if (!dpart)
		return NULL;
	int rc = dsos_part_spec_from_part(&dpart->spec, part);
	if (rc) {
		errno = rc;
		free(dpart);
		return NULL;
	}
	dpart->part = sos_part_get(part);
	dpart->handle = get_next_handle();
	dpart->spec.id = dpart->handle;

	pthread_mutex_lock(&client->part_tree_lock);
	/* Id tree */
	ods_rbn_init(&dpart->id_rbn, &dpart->handle);
	ods_rbt_ins(&client->part_id_tree, &dpart->id_rbn);
	/* Name tree */
	ods_rbn_init(&dpart->name_rbn, (char *)name);
	ods_rbt_ins(&client->part_name_tree, &dpart->name_rbn);
	/* UUID tree */
	ods_rbn_init(&dpart->uuid_rbn, uuid);
	ods_rbt_ins(&client->part_uuid_tree, &dpart->uuid_rbn);
	pthread_mutex_unlock(&client->part_tree_lock);

	return dpart;
}

static const char *perm_mask_to_str(uint32_t mask)
{
	static char s_[16];
	char *s;
 	static struct xlat_perm_s {
		 int bit;
		 char c;
	} translate[] = {
		{ 0001, 'x' },
        	{ 0002, 'w' },
		{ 0004, 'r' },
		{ 0010, 'x' },
		{ 0020, 'w' },
		{ 0040, 'r' },
		{ 0100, 'x' },
		{ 0200, 'w' },
		{ 0400, 'r' }
	};
	struct xlat_perm_s *x;
	int i;
	s = s_;
	for (i = (sizeof(translate)/sizeof(translate[0])); i; i--) {
		x = &translate[i];
		if (0 != (x->bit & mask))
                	*s = x->c;
		else
			*s = '-';
		s++;
	}
	*s = '\0';
	return s_;
}

bool_t part_create_1_svc(dsos_container_id cont_id, dsos_part_spec spec,
			 dsos_part_res *res, struct svc_req *rqst)
{
	char err_msg[256];
	struct dsos_session *client;
	struct dsos_part *dpart;
	sos_part_t part;
	int rc;

	if (!authenticate_request(rqst, __func__))
		return FALSE;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		sprintf(err_msg, "The container %ld does not exist.", cont_id);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	/* Create the partition */
	part = sos_part_open(spec.path,
			SOS_PERM_RW | SOS_PERM_CREAT,
			spec.perm, spec.desc);
	if (!part) {
		res->error = errno ? errno : EACCES;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d creating the partition %s", errno, spec.path);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	/* Apply the uid/gid from the request */
	rc = sos_part_chown(part, spec.user_id, spec.group_id);
	if (rc) {
		res->error = rc;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d changing the owner/group to %ld:%ld on partition %s",
			rc, spec.user_id, spec.group_id, spec.name);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	/* Apply the permissions from the request */
	rc = sos_part_chmod(part, spec.perm);
	if (rc) {
		res->error = rc;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d changing the permissions to %s on partition '%s'",
			rc, perm_mask_to_str(spec.perm), spec.name);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	/* Attach the partition to the container */
	rc = sos_part_attach(client->sos, spec.name, spec.path);
	if (rc) {
		res->error = rc;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d attaching the partition %s to the container as %s",
			rc, spec.path, spec.name);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	/* Put our naked partition handle and look up the one from the container */
	sos_part_put(part);
	part = sos_part_by_name(client->sos, spec.name);
	if (!part) {
		res->error = errno;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d looking up the partition '%s' in the container",
			rc, spec.name);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	/* Make the partition active */
	rc = sos_part_state_set(part, SOS_PART_STATE_ACTIVE);
	if (rc) {
		res->error = rc;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d setting the partition state to ACTIVE on the partition %s",
			rc, spec.name);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_2;
	}
	/* Create and cache the DSOS partition */
	dpart = cache_part(client, part);
	if (!dpart) {
		res->error = errno ? errno : EINVAL;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d caching the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto  out_2;
	}
	rc = dsos_part_spec_from_part(&res->dsos_part_res_u.spec, part);
	if (rc) {
		res->error = errno ? errno : ENOMEM;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d encoding the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto  out_2;
	}
	res->dsos_part_res_u.spec.id = dpart->handle;
	res->error = 0;
out_2:
	sos_part_put(part);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_find_by_id_1_svc(dsos_container_id cont_id, dsos_part_id part_id,
			     dsos_part_res *res, struct svc_req *rqst)
{
	char err_msg[256];
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	memset(res, 0, sizeof(*res));
	res->error = ENOSYS;
	sprintf(err_msg, "This interface is deprecated.");
	res->dsos_part_res_u.error_msg = strdup(err_msg);
	return TRUE;
}

bool_t part_find_by_name_1_svc(dsos_container_id cont_id, char *name,
			       dsos_part_res *res, struct svc_req *rqst)
{
	char err_msg[256];
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_part *dpart;
	sos_part_t part;
	int rc;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		sprintf(err_msg, "The container %ld does not exist.", cont_id);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}
	rbn = ods_rbt_find(&client->part_name_tree, name);
	if (rbn) {
		res->error = 0;
		dpart = container_of(rbn, struct dsos_part, name_rbn);
		goto copy_spec;
	}
	part = sos_part_by_name(client->sos, name);
	if (!part) {
		res->error = errno ? errno : ENOENT;
		snprintf(err_msg, sizeof(err_msg),
			"The partition '%s' does not exist in container %ld",
			name, cont_id);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	dpart = cache_part(client, part);
	sos_part_put(part);	/* cache part takes it's own reference */
	if (!dpart) {
		res->error = errno ? errno : EINVAL;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d caching the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
copy_spec:
	rc = dsos_part_spec_copy(&res->dsos_part_res_u.spec, &dpart->spec);
	if (rc) {
		res->error = errno ? errno : ENOMEM;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d encoding the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	res->dsos_part_res_u.spec.id = dpart->handle;
	res->error = 0;
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_find_by_uuid_1_svc(dsos_container_id cont_id, char *uuid,
			       dsos_part_res *res, struct svc_req *rqst)
{
	char err_msg[256];
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_part *dpart;
	sos_part_t part;
	int rc;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		sprintf(err_msg, "The container %ld does not exist.", cont_id);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_0;
	}

	rbn = ods_rbt_find(&client->part_uuid_tree, uuid);
	if (rbn) {
		res->error = 0;
		dpart = container_of(rbn, struct dsos_part, uuid_rbn);
		goto copy_spec;
	}
	part = sos_part_by_uuid(client->sos, (unsigned char *)uuid);
	if (!part) {
		res->error = errno ? errno : ENOENT;
		snprintf(err_msg, sizeof(err_msg),
			"The partition with UUID '%s' does not exist in container %ld",
			 uuid, cont_id);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	dpart = cache_part(client, part);
	sos_part_put(part);	/* cache part takes it's own reference */
	if (!dpart) {
		res->error = errno ? errno : EINVAL;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d caching the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
copy_spec:
	rc = dsos_part_spec_copy(&res->dsos_part_res_u.spec, &dpart->spec);
	if (rc) {
		res->error = errno ? errno : ENOMEM;
		snprintf(err_msg, sizeof(err_msg),
			"Error %d encoding the partition", res->error);
		res->dsos_part_res_u.error_msg = strdup(err_msg);
		goto out_1;
	}
	res->dsos_part_res_u.spec.id = dpart->handle;
	res->error = 0;
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_state_set_1_svc(dsos_container_id cont_id, dsos_part_id part_id, long part_state,
			    int *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_part *dpart;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		*res = EINVAL;
		goto out_0;
	}
	rbn = ods_rbt_find(&client->part_id_tree, &part_id);
	if (!rbn) {
		*res = ENOENT;
		goto out_1;
	}
	dpart = container_of(rbn, struct dsos_part, id_rbn);
	*res = sos_part_state_set(dpart->part, part_state);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_chown_1_svc(dsos_container_id cont_id, dsos_part_id part_id,
			long uid, long gid, int *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_part *dpart;

	client = get_client(cont_id);
	if (!client) {
		*res = EINVAL;
		goto out_0;
	}
	rbn = ods_rbt_find(&client->part_id_tree, &part_id);
	if (!rbn) {
		*res = ENOENT;
		goto out_1;
	}
	dpart = container_of(rbn, struct dsos_part, id_rbn);
	*res = sos_part_chown(dpart->part, uid, gid);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_chmod_1_svc(dsos_container_id cont_id, dsos_part_id part_id,
			long mode, int *res, struct svc_req *rqst)
{
	if (!authenticate_request(rqst, __func__))
		return FALSE;
	struct dsos_session *client;
	struct ods_rbn *rbn;
	struct dsos_part *dpart;

	memset(res, 0, sizeof(*res));
	client = get_client(cont_id);
	if (!client) {
		*res = EINVAL;
		goto out_0;
	}

	rbn = ods_rbt_find(&client->part_id_tree, &part_id);
	if (!rbn) {
		*res = ENOENT;
		goto out_1;
	}
	dpart = container_of(rbn, struct dsos_part, id_rbn);
	*res = sos_part_chmod(dpart->part, mode);
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t part_query_1_svc(dsos_container_id cont_id, dsos_part_query_res *res, struct svc_req *rqst)
{
	struct dsos_session *client;
	int count, array_size;
	sos_part_t part;

	if (!authenticate_request(rqst, __func__))
		return FALSE;

	client = get_client(cont_id);
	if (!client) {
		res->error = DSOS_ERR_CLIENT;
		goto out_0;
	}

	array_size = 0;
	sos_part_iter_t iter = sos_part_iter_new(client->sos);
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		sos_part_put(part);
		array_size += 1;
	}
	res->dsos_part_query_res_u.names.names_val = calloc(array_size, sizeof(char *));
	res->dsos_part_query_res_u.names.names_len = array_size;
	res->error = 0;
	count = 0;
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		res->dsos_part_query_res_u.names.names_val[count] = strdup(sos_part_name(part));
		sos_part_put(part);
		count += 1;
	}
	sos_part_iter_free(iter);
out_0:
	return TRUE;
}

bool_t obj_create_1_svc(dsos_obj_array obj_array, dsos_obj_create_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_schema *schema;
	dsos_obj_entry *obj_e;

	obj_e = obj_array;
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
		sos_obj_t obj =
			sos_obj_new_from_data(
				schema->schema,
				obj_e->value.dsos_obj_value_val,
				obj_e->value.dsos_obj_value_len);
		if (!obj) {
			res->error = errno;
			goto out_2;
		}
		struct dsos_part *part = NULL;
		if (obj_e->part_id)
			part = get_part(client, obj_e->part_id);
		sos_obj_commit_part(obj, part ? part->part : NULL);
		put_part(part);
		sos_obj_index(obj);
		sos_obj_put(obj);
		obj_e = obj_e->next;
		res->error = 0;
		res->dsos_obj_create_res_u.obj_id = 0; // object->handle;
	}
out_2:
	put_schema(schema);
out_1:
	put_client(client);
out_0:
	pthread_mutex_unlock(&client_tree_lock);
	return TRUE;
}

bool_t obj_delete_1_svc(dsos_obj_array obj_array, int *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	*res = ENOTSUP;
	return TRUE;
}

bool_t obj_update_1_svc(dsos_obj_array obj_array, int *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	*res = ENOTSUP;
	return TRUE;
}

#define DSOS_ITER_OBJ_COUNT	4096
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
	diter->count = DSOS_ITER_OBJ_COUNT;
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

static int __make_obj_array(dsos_obj_array_res *result, struct dsos_iter *diter)
{
	int count;
	int rc = 0;
	struct dsos_obj_entry *entry = NULL;
	memset(result, 0, sizeof(*result));
	result->dsos_obj_array_res_u.obj_array.obj_array_len = 0;
	result->dsos_obj_array_res_u.obj_array.obj_array_val =
		calloc(diter->count, sizeof(*entry));
	assert(result->dsos_obj_array_res_u.obj_array.obj_array_val);
	result->error = DSOS_ERR_ITER_EMPTY;
	count = 0;
	while (!rc && count < diter->count) {
		result->error = 0;
		sos_obj_t obj = sos_iter_obj(diter->iter);
		if (!obj) {
			result->error = errno;
			goto err_0;
		}
		entry = &result->dsos_obj_array_res_u.obj_array.obj_array_val[count];
		entry->next = NULL;
		result->dsos_obj_array_res_u.obj_array.obj_array_len = count + 1;

		entry->cont_id = diter->cont_id;
		entry->schema_id = diter->schema_id;

		void *obj_data = sos_obj_ptr(obj);
		entry->value.dsos_obj_value_len = sos_obj_size(obj);
		entry->value.dsos_obj_value_val = malloc(entry->value.dsos_obj_value_len);
		memcpy(entry->value.dsos_obj_value_val, obj_data, entry->value.dsos_obj_value_len);
		sos_obj_put(obj);
		count ++;
		if (count < diter->count)
			rc = sos_iter_next(diter->iter);
	}
	return 0;
err_0:
	return -1;
}

bool_t iter_begin_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_array_res *res, struct svc_req *req)
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
		rc = __make_obj_array(res, diter);
	} else {
		res->error = rc;
	}
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_end_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_array_res *res, struct svc_req *req)
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

	rc = sos_iter_end(diter->iter);
	if (!rc) {
		rc = __make_obj_array(res, diter);
	} else {
		res->error = rc;
	}
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_next_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_obj_array_res *res, struct svc_req *req)
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
		rc = __make_obj_array(res, diter);
	} else {
		res->error = rc;
	}
out_1:
	put_client(client);
out_0:
	return TRUE;
}

bool_t iter_prev_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_obj_array_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_bytes bytes, dsos_obj_array_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_glb_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_bytes bytes, dsos_obj_array_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}
bool_t iter_find_lub_1_svc(dsos_container_id cont, dsos_iter_id iter, dsos_bytes bytes, dsos_obj_array_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	return TRUE;
}

bool_t iter_stats_1_svc(dsos_container_id cont, dsos_iter_id iter_id, dsos_iter_stats_res *res, struct svc_req *req)
{
	if (!authenticate_request(req, __func__))
		return FALSE;
	struct dsos_session *client;
	struct dsos_iter *diter;
	int rc;
	memset(res, 0, sizeof(*res));
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

	res->dsos_iter_stats_res_u.stats.cardinality = sos_iter_card(diter->iter);
	res->dsos_iter_stats_res_u.stats.duplicates = sos_iter_dups(diter->iter);
	res->dsos_iter_stats_res_u.stats.size_bytes = sos_iter_size(diter->iter);
 out_1:
	put_client(client);
 out_0:
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

bool_t query_select_1_svc(dsos_container_id cont_id, dsos_query_id query_id,
			  dsos_query query_str, dsos_query_select_res *res,
			  struct svc_req *rqst)
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

struct bin_tree_entry {
	sos_key_t bin_k;	/* key for the bin */
	sos_obj_t obj;		/* object in this bin */
	uint32_t count;		/* Number of objects in this bin */
	struct ods_rbn rbn;
};

static int resample_object(struct ast *ast, sos_obj_t obj)
{
	SOS_KEY(res_key);
	sos_key_t iter_key = sos_iter_key(ast->sos_iter); /* This is the source object key */
	sos_attr_t iter_attr = ast->iter_attr_e->sos_attr;
	uint32_t timestamp, remainder;
	size_t count;
	sos_value_t join_attrs;
	struct ods_rbn *rbn;
	int rc, id, i;
	rc = sos_key_copy(res_key, iter_key);
	if (!rc)
		goto out;
	sos_comp_key_spec_t res_spec = sos_comp_key_get(res_key, &count);

	/* Create a bin-key from the object */
	switch (sos_attr_type(ast->iter_attr_e->sos_attr)) {
	case SOS_TYPE_JOIN:
		for (i = 0; i < count; i++) {
			if (res_spec[i].type != SOS_TYPE_TIMESTAMP)
				continue;
			timestamp = res_spec[i].data->prim.timestamp_.fine.secs;
			remainder = timestamp % (uint32_t)ast->bin_width;
			timestamp -= remainder;
			res_spec[i].data->prim.timestamp_.fine.secs = timestamp;
			res_spec[i].data->prim.timestamp_.fine.usecs = 0;
		}
		rc = sos_comp_key_set(res_key, count, res_spec);
		sos_comp_key_free(res_spec, count);
		break;
	case SOS_TYPE_TIMESTAMP:
		break;
	default:
		assert(0 == "Invalid resample bin key type");
	}
	/* Find the bin for this object */
	rbn = ods_rbt_find(&ast->bin_tree, res_key);
	if (rbn) {
		struct bin_tree_entry *be = container_of(rbn, struct bin_tree_entry, rbn);
		assert(be->obj);
		sos_key_put(res_key);
		be->count += 1;
		/* Average the values of all numeric attributes in the object */
		sos_obj_put(obj);
	} else {
		struct bin_tree_entry *be = calloc(1, sizeof(*be));
		be->bin_k = sos_key_new(sos_key_size(res_key));
		sos_key_copy(be->bin_k, res_key);
		sos_key_put(res_key);
		ods_rbn_init(&be->rbn, be->bin_k);
		sos_obj_t result_obj = sos_obj_malloc(ast->result_schema);
		ast_attr_entry_t attr_e;
		TAILQ_FOREACH(attr_e, &ast->select_list, link) {
			sos_obj_attr_copy(result_obj, attr_e->res_attr, obj, attr_e->sos_attr);
		}
		be->obj = result_obj;
		be->count = 1;
		ods_rbt_ins(&ast->bin_tree, &be->rbn);
	}
out:
	return rc;
}

#define QUERY_OBJECT_COUNT	(65536)
static int __make_query_obj_array(struct dsos_session *client, struct ast *ast,
				  dsos_query_next_res *result)
{
	enum ast_eval_e eval;
	sos_iter_t iter = ast->sos_iter;
	int count = QUERY_OBJECT_COUNT;
	sos_obj_t obj;
	struct dsos_obj_entry *entry = NULL;
	int obj_id, rc = 0;
	ast_attr_entry_t attr_e;

	memset(result, 0, sizeof(*result));
	result->error = DSOS_ERR_QUERY_EMPTY;
	result->dsos_query_next_res_u.result.obj_array.obj_array_val =
		calloc(count, sizeof(dsos_obj_entry));

	obj_id = 0;
	while (!rc && count) {
		obj = sos_iter_obj(iter);
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
		struct dsos_part *dpart;
		sos_part_t part = sos_obj_part(obj);
		uuid_t uuid;
		sos_part_uuid(part, uuid);
		struct ods_rbn *rbn = ods_rbt_find(&client->part_uuid_tree, uuid);
		if (rbn) {
			dpart = container_of(rbn, struct dsos_part, uuid_rbn);
		} else {
			dpart = cache_part(client, part);
			if (!dpart)
				goto out;
		}

		result->error = 0;
		entry = &result->dsos_query_next_res_u.result.obj_array.obj_array_val[obj_id++];
		result->dsos_query_next_res_u.result.obj_array.obj_array_len += 1;
		result->dsos_query_next_res_u.result.format = 0;
		entry->next = NULL;
		entry->cont_id = client->handle;
		entry->part_id = dpart->spec.id;
		entry->schema_id = 0;
		sos_obj_ref_t ref = sos_obj_ref(obj);
		entry->obj_ref = ref.ref.obj;
		count --;
		void *obj_data = sos_obj_ptr(result_obj);
		entry->value.dsos_obj_value_len = sos_obj_size(result_obj);
		entry->value.dsos_obj_value_val = malloc(entry->value.dsos_obj_value_len);
		memcpy(entry->value.dsos_obj_value_val, obj_data, entry->value.dsos_obj_value_len);
		sos_obj_put(obj);
		sos_obj_put(result_obj);
		if (count)
			rc = sos_iter_next(iter);
		if (ast->result_limit && ast->result_count >= ast->result_limit)
			rc = ENOENT;
	}
out:
	return result->error;
err_0:
	return -1;
}

static int __make_query_obj_bins(struct dsos_session *client, struct ast *ast,
				  dsos_query_next_res *result)
{
	enum ast_eval_e eval;
	sos_iter_t iter = ast->sos_iter;
	int count = QUERY_OBJECT_COUNT;
	sos_obj_t obj;
	struct dsos_obj_entry *entry = NULL;
	int obj_id, rc = 0;
	ast_attr_entry_t attr_e;

	memset(result, 0, sizeof(*result));
	result->error = DSOS_ERR_QUERY_EMPTY;
	result->dsos_query_next_res_u.result.obj_array.obj_array_val =
		calloc(count, sizeof(dsos_obj_entry));

	while (!rc && count) {
		obj = sos_iter_obj(iter);
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
		rc = resample_object(ast, obj);
		if (!rc)
			rc = sos_iter_next(iter);
	}
out:
	obj_id = 0;
	result->error = 0;
	while (!ods_rbt_empty(&ast->bin_tree)) {
		struct ods_rbn *rbn = ods_rbt_min(&ast->bin_tree);
		struct bin_tree_entry *be = container_of(rbn, struct bin_tree_entry, rbn);
		sos_obj_ref_t ref;
		void *obj_data;
		ods_rbt_del(&ast->bin_tree, rbn);
		entry = &result->dsos_query_next_res_u.result.obj_array.obj_array_val[obj_id++];
		result->dsos_query_next_res_u.result.obj_array.obj_array_len += 1;
		result->dsos_query_next_res_u.result.format = 0;
		entry->next = NULL;
		entry->cont_id = client->handle;
		entry->part_id = 0;
		entry->schema_id = 0;
		ref = sos_obj_ref(be->obj);
		entry->obj_ref = ref.ref.obj;
		obj_data = sos_obj_ptr(be->obj);
		entry->value.dsos_obj_value_len = sos_obj_size(be->obj);
		entry->value.dsos_obj_value_val = malloc(entry->value.dsos_obj_value_len);
		memcpy(entry->value.dsos_obj_value_val, obj_data, entry->value.dsos_obj_value_len);
		sos_key_put(be->bin_k);
		sos_obj_put(be->obj);
		free(be);
	}
	return result->error;
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

	if (query->ast->result_limit && query->ast->result_count >= query->ast->result_limit)
		goto empty;

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
		if (query->ast->bin_width)
			rc = __make_query_obj_bins(client, query->ast, res);
		else
			rc = __make_query_obj_array(client, query->ast, res);
		if (!rc)
			query->state = DSOSQ_STATE_NEXT;
		else
			goto empty;
		break;
	case DSOSQ_STATE_NEXT:
		rc = sos_iter_next(query->ast->sos_iter);
		if (rc)
			goto empty;
		if (query->ast->bin_width)
			rc = __make_query_obj_bins(client, query->ast, res);
		else
			rc = __make_query_obj_array(client, query->ast, res);
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
