
#define _GNU_SOURCE
#include <sys/queue.h>
#include <sos/sos.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include "sosapi.h"
#include "sosapi_common.h"
#include "dsos.h"

typedef struct dsos_client_request_s {
	TAILQ_ENTRY(dsos_client_request_s) link;
	dsos_container_id cont_id;
	enum dsos_client_request_type_e {
		REQ_TRANSACTION_BEGIN = 1,
		REQ_OBJ_CREATE = 2,
	} type;
	union {
		struct transaction_begin_s {
			dsos_container_id cont_id;
		} transaction_begin;
		struct create_s {
			dsos_obj_entry *obj_entry;
		} obj_create;
	} request;
} *dsos_client_request_t;

typedef struct dsos_client_s {
	CLIENT *client;
	pthread_mutex_t rpc_lock;
	int client_id;
	int shutdown;
	pthread_t thread;
	int queue_depth;
	double obj_count;
	double request_count;
	/* Requests queued here between x_start and x_end */
	pthread_mutex_t queue_lock;
	pthread_cond_t queue_cond;
	TAILQ_HEAD(client_req_queue_q, dsos_client_request_s) queue_q;
	/* Requests flushed from here between x_end and x_start */
	pthread_mutex_t flush_lock;
	pthread_cond_t flush_cond;
 	TAILQ_HEAD(client_req_flush_q, dsos_client_request_s) flush_q;
} *dsos_client_t;

struct dsos_session_s {
	int client_count;	/* Number of clients */
	int next_client;	/* The next client to use */
	const char **hosts;	/* One for each client */
	struct dsos_client_s *clients;
	enum dsos_session_state_e {
		CONNECTED,
		DISCONNECTED,
		ERROR
	} state;
};

struct dsos_container_s {
	char *path;
	dsos_session_t sess;	/* The session */
	int handle_count;	/* Number of servers */
	int handles[];		/* Array of container handles from each server */
};

struct dsos_schema_s {
	dsos_container_t cont;	/* The container */
	sos_schema_t schema;	/* The SOS schema */
	int handles[];		/* Array of schema handles from each server */
};

typedef struct dsos_obj_ref_s {
	int client_id;			/* Client this object came from */
	struct sos_value_s key_value;	/* Iterator key value */
	sos_obj_t obj;			/* SOS object */
	struct ods_rbn rbn;		/* obj_tree node */
} *dsos_obj_ref_t;

struct dsos_iter_s {
	enum iter_action {
		DSOS_ITER_BEGIN,
		DSOS_ITER_END,
		DSOS_ITER_NEXT,
		DSOS_ITER_PREV
	} action;
	dsos_container_t cont;		/* The container */
	dsos_schema_t schema;		/* The schema for objects on this iterator */
	sos_attr_t key_attr;		/* The attribute that is the key */
	dsos_iter_id *handles;		/* Server specific handle */
	int *counts;				/* Objects in tree from each client */
	struct ods_rbt obj_tree;	/* RBT of objects indexed by key_attr */
};

struct dsos_query_s {
	enum query_state {
		DSOS_QUERY_INIT,	/* Created, no select */
		DSOS_QUERY_SELECT,	/* Select completed, ready to deliver data */
		DSOS_QUERY_NEXT,	/* query_next has been called */
		DSOS_QUERY_EMPTY	/* No more data */
	} state;
	dsos_container_t cont;		/* The container */
	dsos_query_id *handles;		/* Server specific handle */
	int *counts;				/* Objects in tree from each client */
	sos_schema_t schema;		/* The schema for objects on this iterator */
	sos_attr_t key_attr;		/* The attribute that is the key */
	struct ods_rbt obj_tree;	/* RBT of objects indexed by key_attr */
};

struct dsos_session_s g_session;

static inline void dsos_res_init(dsos_container_t cont, dsos_res_t *res) {
	int i;
	res->count = cont->handle_count;
	for (i = 0; i < res->count; i++)
		res->res[i] = 0;
	res->any_err = 0;
}

static int handle_transaction_begin(dsos_client_t client, dsos_client_request_t rqst)
{
	int rres;
	enum clnt_stat rpc_err;

	// fprintf(stderr, "Beginning transaction on client %d\n", client->client_id);
	pthread_mutex_lock(&client->rpc_lock);
	rpc_err = transaction_begin_1(rqst->cont_id, &rres, client->client);
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS) {
		fprintf(stderr, "transaction_begin_1 failed on client %d with RPC error %d\n",
			client->client_id, rpc_err);
	}
	if (rres) {
		fprintf(stderr, "transaction_begin_1 failed on client %d with error %d\n",
			client->client_id, rres);
	}
	free(rqst);
	return 0;
}

static int handle_obj_create(dsos_client_t client, dsos_client_request_t rqst)
{
	dsos_create_res create_res = {};

	// fprintf(stderr, "Creating object on client %d\n", client->client_id);
	pthread_mutex_lock(&client->rpc_lock);
	enum clnt_stat rpc_err = obj_create_1(rqst->request.obj_create.obj_entry, &create_res, client->client);
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS) {
		fprintf(stderr, "obj_create_1 failed with RPC error %d\n", rpc_err);
		return rpc_err;
	}
	if (create_res.error)
		fprintf(stderr, "obj_create_1 returned error %d\n", create_res.error);
	dsos_obj_entry *obj_e = rqst->request.obj_create.obj_entry;
	while (obj_e) {
		dsos_obj_entry *next_obj_e = obj_e->next;
		free(obj_e->value.dsos_obj_value_val);
		free(obj_e);
		obj_e = next_obj_e;
	}
	free(rqst);
	return create_res.error;
}

/*
 * Processes the flush queue for a DSOS RPC client
 */
void *client_proc_fn(void *arg)
{
	dsos_client_t client = arg;
	dsos_client_request_t rqst;
	struct timespec timeout;
next:
	pthread_mutex_lock(&client->flush_lock);
	while (TAILQ_EMPTY(&client->flush_q)) {
		if (client->shutdown)
			return NULL;
		timeout.tv_sec = time(NULL) + 1;	/* This timeout is short to allow for checking client->shutdown */
		timeout.tv_nsec = 0;
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
		int rc = pthread_cond_timedwait(&client->flush_cond, &client->flush_lock, &timeout);
		if (rc && rc != ETIMEDOUT)
			fprintf(stderr,
				"Error %d waiting for queue condition "
				"variable on client %d\n",
				rc, client->client_id);
	}
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	while (!TAILQ_EMPTY(&client->flush_q)) {
		dsos_container_id cont_id;
		rqst = TAILQ_FIRST(&client->flush_q);
		TAILQ_REMOVE(&client->flush_q, rqst, link);
		client->queue_depth -= 1;

		switch (rqst->type) {
		case REQ_OBJ_CREATE:
			cont_id = rqst->request.obj_create.obj_entry->cont_id;
			handle_obj_create(client, rqst);
			break;
		case REQ_TRANSACTION_BEGIN:
			cont_id = rqst->request.transaction_begin.cont_id;
			handle_transaction_begin(client, rqst);
			break;
		default:
			assert(0 == "Invalid request type");
		};

		/*
		 * If this was the last request in the flush queue, complete
		 * the transaction
		 */
		// TODO: it's possible that requests refer to different containers. This code does not handle that.
		if (TAILQ_EMPTY(&client->flush_q)) {
			int rres;
			pthread_mutex_lock(&client->rpc_lock);
			enum clnt_stat rpc_err = transaction_end_1(cont_id, &rres, client->client);
			pthread_mutex_unlock(&client->rpc_lock);
			if (rpc_err != RPC_SUCCESS) {
				fprintf(stderr, "transaction_end_1 failed on client %d with RPC error %d\n",
				client->client_id, rpc_err);
			}
			if (rres) {
				fprintf(stderr, "transaction_begin_1 failed on client %d with error %d\n",
					client->client_id, rres);
			}
			// fprintf(stderr, "Ending transaction on client %d\n", client->client_id);
		}
	}
	pthread_mutex_unlock(&client->flush_lock);
	goto next;

	return NULL;
}

void dsos_session_close(dsos_session_t sess)
{
	dsos_client_t client;
	int client_id;

	for (client_id = 0; client_id < sess->client_count; client_id ++) {
		client = &sess->clients[client_id];
		client->shutdown = 1;
	}
	for (client_id = 0; client_id < sess->client_count; client_id ++) {
		client = &sess->clients[client_id];
		void *dontcare;
		pthread_join(client->thread, &dontcare);
		auth_destroy(client->client->cl_auth);
		clnt_destroy(client->client);
	}
}
dsos_session_t dsos_session_open(const char *config_file)
{
	dsos_session_t session;
	char hostname[256];
	char *s;
	int i;
	int host_count;
	FILE* f = fopen(config_file, "r");
	if (!f)
		return NULL;
	host_count = 0;
	while (NULL != (s = fgets(hostname, sizeof(hostname), f))) {
		host_count += 1;
	}
	session = calloc(1, sizeof(*session));
	if (!session)
		return NULL;
	session->client_count = host_count;
	session->hosts = calloc(host_count, sizeof(void *));
	session->clients = calloc(host_count, sizeof(struct dsos_client_s));

	i = 0;
	fseek(f, 0L, SEEK_SET);
	while (NULL != (s = fgets(hostname, sizeof(hostname), f))) {
		/* Strip the newline if present */
		char *s = strstr(hostname, "\n");
		if (s)
			*s = '\0';
		session->hosts[i] = strdup(hostname);
		i += 1;
	}

	for (i = 0; i < host_count; i++) {
		CLIENT *clnt = clnt_create(session->hosts[i], SOSDB, SOSVERS, "tcp");
		if (clnt == NULL) {
			fprintf(stderr, "Error creating client %d\n", i);
			exit (1);
		}
		struct timeval timeout = {
			600, 0
		};
		clnt_control(clnt, CLSET_TIMEOUT, (char *)&timeout);
		dsos_client_t client = &session->clients[i];
		pthread_mutex_init(&client->rpc_lock, NULL);
		client->client = clnt;
		client->client_id = i;
		client->shutdown = 0;
		client->request_count = 0;
		client->queue_depth = 0;
		client->obj_count = 0;
		TAILQ_INIT(&client->queue_q);
		TAILQ_INIT(&client->flush_q);
		pthread_mutex_init(&client->queue_lock, NULL);
		pthread_cond_init(&client->queue_cond, NULL);
		pthread_mutex_init(&client->queue_lock, NULL);
		pthread_cond_init(&client->flush_cond, NULL);
		int rc = pthread_create(&client->thread, NULL, client_proc_fn, client);
		if (rc) {
			return NULL;
		}
		char thread_name[16];
		sprintf(thread_name, "client:%d", i);
		pthread_setname_np(client->thread, thread_name);
	}

	session->next_client = 0;
	return session;
}

void dsos_container_close(dsos_container_t cont)
{
	;
}
void dsos_container_commit(dsos_container_t cont)
{
	;
}

dsos_container_t
dsos_container_open(
		dsos_session_t sess,
		const char *path,
		sos_perm_t perm,
		int mode)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_container_t cont = calloc(1, sizeof *cont + sess->client_count * sizeof(int *));
	if (!cont)
		return NULL;
	cont->sess = sess;
	cont->handle_count = sess->client_count;
	cont->path = strdup(path);
	if (!cont->path)
		goto err_0;

	dsos_open_res open_res;
	for (i = 0; i < cont->handle_count; i++) {
		/* Open/Create the container */
		dsos_client_t client = &sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = open_1((char *)path, SOS_PERM_RW | SOS_PERM_CREAT, 0660, &open_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "open_1 failed with RPC error %d\n", rpc_err);
			goto err_1;
		}
		if (open_res.error) {
			fprintf(stderr, "open_1 failed on client %d with error %d\n", i, open_res.error);
			goto err_1;
		}
		cont->handles[i] = open_res.dsos_open_res_u.cont;
		assert(cont->handles[i]);
	}
	return cont;
err_1:
	for (i = 0; i < sess->client_count; i++) {
		int rc;
		if (cont->handles[i]) {
			dsos_client_t client = &sess->clients[i];
			pthread_mutex_lock(&client->rpc_lock);
			rpc_err = close_1(cont->handles[i], &rc, client->client);
			pthread_mutex_unlock(&client->rpc_lock);
		}
	}
	free(cont->path);
err_0:
	free(cont);
	return NULL;
}

static inline dsos_schema_t dsos_schema_alloc(dsos_container_t cont)
{
	dsos_schema_t s = malloc(sizeof(*s) + (cont->handle_count * sizeof(s->handles[0])));
	if (s)
		s->cont = cont;
	return s;
}

static inline void dsos_schema_free(dsos_schema_t s)
{
	free(s);
}

dsos_schema_t dsos_schema_create(dsos_container_t cont, sos_schema_t schema, dsos_res_t *res)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_schema_t dschema = dsos_schema_alloc(cont);
	dsos_res_init(cont, res);
	dsos_schema_spec *spec = dsos_spec_from_schema(schema);
	if (!spec) {
		res->any_err = errno;
		return NULL;
	}
	dsos_schema_res schema_res = {};
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = schema_create_1(cont->handles[i], *spec, &schema_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "schema_create_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			res->any_err = DSOS_ERR_CLIENT;
			break;
		}
		if (schema_res.error) {
			fprintf(stderr, "schema_create_1 failed on client %d with error %d\n",
				i, rpc_err);
			res->any_err = schema_res.error;
			xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&schema_res);
			break;
		}
		if (i == 0) {
			dschema->schema = dsos_schema_from_spec(schema_res.dsos_schema_res_u.spec);
			assert(dschema->schema);
		}
		dschema->handles[i] = schema_res.dsos_schema_res_u.spec->id;
		xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&schema_res);
	}
	if (res->any_err) {
		dsos_schema_free(dschema);
		dschema = NULL;
	}
	dsos_spec_free(spec);
	return dschema;
}

dsos_schema_t dsos_schema_by_name(dsos_container_t cont, const char *name,
				  dsos_res_t *res)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_schema_res schema_res = {};
	dsos_schema_t schema;
	dsos_res_init(cont, res);

	schema = dsos_schema_alloc(cont);
	if (!schema) {
		res->any_err = ENOMEM;
		return NULL;
	}
	schema->schema = NULL;
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = schema_find_by_name_1(cont->handles[i], (char *)name, &schema_res,
					client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "schema_find_by_name_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			res->any_err = DSOS_ERR_CLIENT;
			break;
		}
		if (schema_res.error) {
			res->any_err = schema_res.error;
			fprintf(stderr, "schema_find_by_name_1 failed on client %d with error %d\n",
				i, schema_res.error);
			break;
		}
		if (i == 0) {
			/* We only need to instantiate one local instance of the schema */
			schema->schema = dsos_schema_from_spec(schema_res.dsos_schema_res_u.spec);
			assert(schema->schema);
		}
		schema->handles[i] = schema_res.dsos_schema_res_u.spec->id;
		xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&schema_res);
	}
	res->res[i] = res->any_err;
	if (res->any_err) {
		sos_schema_free(schema->schema);
		free(schema);
		schema = NULL;
	}
	return schema;
}

dsos_schema_t dsos_schema_by_uuid(dsos_container_t cont, uuid_t uuid,
				  dsos_res_t *res)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_schema_res schema_res = {};
	dsos_schema_t schema;
	dsos_res_init(cont, res);

	schema = dsos_schema_alloc(cont);
	if (!schema) {
		res->any_err = ENOMEM;
		return NULL;
	}
	schema->schema = NULL;
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = schema_find_by_uuid_1(cont->handles[i], (char *)uuid, &schema_res,
					client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "schema_find_by_name_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			res->any_err = DSOS_ERR_CLIENT;
			break;
		}
		if (schema_res.error) {
			res->any_err = schema_res.error;
			fprintf(stderr, "schema_find_by_name_1 failed on client %d with error %d\n",
				i, schema_res.error);
			break;
		}
		if (i == 0) {
			/* We only need to instantiate one local instance of the schema */
			schema->schema = dsos_schema_from_spec(schema_res.dsos_schema_res_u.spec);
			assert(schema->schema);
		}
		schema->handles[i] = schema_res.dsos_schema_res_u.spec->id;
		xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&schema_res);
	}
	res->res[i] = res->any_err;
	if (res->any_err) {
		sos_schema_free(schema->schema);
		free(schema);
		schema = NULL;
	}
	return schema;
}

sos_attr_t dsos_schema_attr_by_id(dsos_schema_t schema, int attr_id)
{
	if (schema->schema)
		return sos_schema_attr_by_id(schema->schema, attr_id);
	return NULL;
}
sos_attr_t dsos_schema_attr_by_name(dsos_schema_t schema, const char *name)
{
	if (schema->schema)
		return sos_schema_attr_by_name(schema->schema, name);
	return NULL;
}

dsos_name_array_t dsos_schema_query(dsos_container_t cont, dsos_res_t *res)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_res_init(cont, res);
	dsos_name_array_t names = NULL;
	dsos_schema_query_res query_res;
	dsos_client_t client = &cont->sess->clients[0];

	pthread_mutex_lock(&client->rpc_lock);
	rpc_err = schema_query_1(cont->handles[0], &query_res, client->client);
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS) {
		res->any_err = DSOS_ERR_CLIENT;
		fprintf(stderr, "schema_query_1 failed on client %d with RPC error %d\n", 0, rpc_err);
	}
	if (query_res.error) {
		res->any_err = query_res.error;
		fprintf(stderr, "schema_query_1 failed on client %d with error %d\n", 0, query_res.error);
	}
	names = calloc(1, sizeof(*names));
	names->count = query_res.dsos_schema_query_res_u.names.names_len;
	names->names = calloc(names->count, sizeof(char *));
	for (i = 0; i < names->count; i++) {
		names->names[i] =
			strdup(query_res.dsos_schema_query_res_u.names.names_val[i]);
	}
	xdr_free((xdrproc_t)xdr_dsos_schema_query_res, (char *)&query_res);
	return names;
}

void dsos_transaction_begin(dsos_container_t cont, dsos_res_t *res)
{
	int client_id;
	dsos_client_t client;
	dsos_res_init(cont, res);
	for (client_id = 0; client_id < cont->handle_count; client_id++) {
		dsos_client_request_t rqst = malloc(sizeof *rqst);
		if (!rqst)
			goto enomem;
		client = &cont->sess->clients[client_id];
		rqst->cont_id = cont->handles[client_id];
		rqst->request.transaction_begin.cont_id = rqst->cont_id;
		rqst->type = REQ_TRANSACTION_BEGIN;
		pthread_mutex_lock(&client->queue_lock);
		TAILQ_INSERT_TAIL(&client->queue_q, rqst, link);
		client->queue_depth += 1;
		client->request_count += 1.0;
		pthread_mutex_unlock(&client->queue_lock);
		pthread_cond_signal(&client->queue_cond);
	}
	return;

enomem:
	res->any_err = errno;
	res->res[client_id] = errno;
	return;

}

void dsos_transaction_end(dsos_container_t cont, dsos_res_t *res)
{
	dsos_client_t client;
	int i;
	dsos_client_request_t rqst;
	dsos_res_init(cont, res);

	for (i = 0; i < cont->handle_count; i++) {
		/* Move all requests to the flush_q */
		client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->queue_lock);
		pthread_mutex_lock(&client->flush_lock);
		while (!TAILQ_EMPTY(&client->queue_q)) {
			rqst = TAILQ_FIRST(&client->queue_q);
			TAILQ_REMOVE(&client->queue_q, rqst, link);
			TAILQ_INSERT_TAIL(&client->flush_q, rqst, link);
			// fprintf(stderr, "%s: moved rqst %p to flush_q on client %d\n", __func__, rqst, i);
		}
		pthread_mutex_unlock(&client->flush_lock);
		pthread_mutex_unlock(&client->queue_lock);
		/* Kick the client thread */
		pthread_cond_signal(&client->flush_cond);
	}
}

sos_obj_t dsos_obj_new(dsos_schema_t schema)
{
	return sos_obj_malloc(schema->schema);
}

void dsos_obj_create(dsos_container_t cont, dsos_schema_t schema, sos_obj_t obj, dsos_res_t *res)
{
	dsos_obj_entry *obj_e;
	dsos_client_t client;
	int client_id;
	dsos_container_id cont_id;
	dsos_res_init(cont, res);

	client_id = __sync_fetch_and_add(&cont->sess->next_client, 1) % cont->sess->client_count;
	client = &cont->sess->clients[client_id];
	cont_id = cont->handles[client_id];
	obj_e = malloc(sizeof *obj_e);
	if (!obj_e)
		goto enomem;

	obj_e->cont_id = cont->handles[client_id];
	obj_e->schema_id = schema->handles[client_id];
	size_t obj_sz = sos_obj_size(obj);
	obj_e->value.dsos_obj_value_len = obj_sz;
	obj_e->value.dsos_obj_value_val = malloc(obj_sz);
	memcpy(obj_e->value.dsos_obj_value_val, sos_obj_ptr(obj), obj_sz);
	obj_e->next = NULL;

	pthread_mutex_lock(&client->queue_lock);
	dsos_client_request_t rqst = TAILQ_LAST(&client->queue_q, client_req_queue_q);
	/*
	 * If there is already an obj create request on the queue, add
	 * this object to the object list for that request
	 */
	client->obj_count += 1.0;
	if (rqst && rqst->type == REQ_OBJ_CREATE && rqst->cont_id == cont_id) {
		obj_e->next = rqst->request.obj_create.obj_entry;
		rqst->request.obj_create.obj_entry = obj_e;
	} else {
		rqst = malloc(sizeof *rqst);
		rqst->cont_id = cont->handles[client_id];
		rqst->type = REQ_OBJ_CREATE;
		rqst->request.obj_create.obj_entry = obj_e;
		TAILQ_INSERT_TAIL(&client->queue_q, rqst, link);
		client->request_count += 1.0;
	}
	pthread_mutex_unlock(&client->queue_lock);
	pthread_cond_broadcast(&client->queue_cond);
	return;

enomem:
	res->any_err = errno;
	res->res[client_id] = errno;
	return;
}

static int64_t key_comparator(void *a, const void *b, void *arg)
{
	return sos_value_cmp((sos_value_t)a, (sos_value_t)b);
}

dsos_iter_t dsos_iter_create(dsos_container_t cont, dsos_schema_t schema, const char *attr_name)
{
	enum clnt_stat rpc_err;
	int i, res;
	sos_attr_t key_attr;
	dsos_iter_res iter_res;
	dsos_iter_t iter;

	key_attr = sos_schema_attr_by_name(schema->schema, attr_name);
	if (!key_attr) {
		errno = ENOENT;
		return NULL;
	}
	if (0 == sos_attr_is_indexed(key_attr)) {
		errno = EINVAL;
		return NULL;
	}
	iter = calloc(1, sizeof *iter);
	if (!iter)
		return NULL;
	iter->key_attr = key_attr;
	iter->cont = cont;
	iter->schema = schema;
	iter->handles = calloc(cont->handle_count, sizeof(dsos_iter_id));
	if (!iter->handles) {
		errno = ENOMEM;
		goto err_0;
	}
	iter->counts = calloc(cont->handle_count, sizeof(int));
	if (!iter->counts) {
		errno = ENOMEM;
		goto err_1;
	}
	ods_rbt_init(&iter->obj_tree, key_comparator, NULL);
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		memset(&iter_res, 0, sizeof(iter_res));
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = iter_create_1(cont->handles[i], schema->handles[i],
				(char *)attr_name, &iter_res,client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "iter_create_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			errno = ENETDOWN;
			goto err_2;
		}
		if (iter_res.error) {
			fprintf(stderr, "iter_create_1 failed on client %d with error %d\n",
				i, iter_res.error);
			errno = iter_res.error;
			goto err_2;
		}
		iter->handles[i] = iter_res.dsos_iter_res_u.iter_id;
	}
	return iter;
err_2:
	for (; i > 0; i--) {
		dsos_client_t client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = iter_delete_1(cont->handles[i-1], iter->handles[i-1],
					&res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
	}
	free(iter->counts);
err_1:
	free(iter->handles);
err_0:
	free(iter);
	return NULL;
}

static int iter_obj_add(dsos_iter_t iter, int client_id)
{
	dsos_container_id cont_id = iter->cont->handles[client_id];
	dsos_iter_id iter_id = iter->handles[client_id];
	enum clnt_stat rpc_err = 0;
	dsos_obj_list_res obj_res;
	dsos_obj_entry *obj_e;
	int count = 0;
	dsos_client_t client = &iter->cont->sess->clients[client_id];
	memset(&obj_res, 0, sizeof(obj_res));
	pthread_mutex_lock(&client->rpc_lock);
	switch (iter->action) {
	case DSOS_ITER_BEGIN:
		rpc_err = iter_begin_1(cont_id, iter_id, &obj_res, client->client);
		break;
	case DSOS_ITER_NEXT:
		rpc_err = iter_next_1(cont_id, iter_id, &obj_res, client->client);
		break;
	case DSOS_ITER_PREV:
		rpc_err = iter_prev_1(cont_id, iter_id, &obj_res, client->client);
		break;
	case DSOS_ITER_END:
		rpc_err = iter_end_1(cont_id, iter_id, &obj_res, client->client);
		break;
	}
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS)
		return RPC_ERROR(rpc_err);
	if (obj_res.error)
		return obj_res.error;

	/* Build the objects from the list */
	obj_e = obj_res.dsos_obj_list_res_u.obj_list;
	while (obj_e) {
		dsos_obj_entry *next_obj_e = obj_e->next;

		dsos_obj_ref_t obj_ref = malloc(sizeof *obj_ref);
		assert(obj_ref);
		obj_ref->client_id = client_id;
		sos_obj_t obj = sos_obj_new_with_data(
					iter->schema->schema,
					obj_e->value.dsos_obj_value_val,
					obj_e->value.dsos_obj_value_len);
		assert(obj);
		obj_ref->obj = obj;
		sos_value_init(&obj_ref->key_value, obj, iter->key_attr);
		ods_rbn_init(&obj_ref->rbn, &obj_ref->key_value);
		ods_rbt_ins(&iter->obj_tree, &obj_ref->rbn);
		iter->counts[client_id] += 1;
		count += 1;
		free(obj_e->value.dsos_obj_value_val);
		free(obj_e);

		obj_e = next_obj_e;
	}
	if (!count)
		return ENOENT;
	return 0;
}

static sos_obj_t iter_obj(dsos_iter_t iter, struct ods_rbn *rbn)
{
	sos_obj_t obj;
	dsos_obj_ref_t ref;
	int rc;

	if (!rbn)
		return NULL;

	ref = container_of(rbn, struct dsos_obj_ref_s, rbn);
	obj = ref->obj;
	ods_rbt_del(&iter->obj_tree, rbn);
	sos_value_put(&ref->key_value);
	/*
	 * If all objects from a given client have been consumed,
	 * replenish the tree with more objects from that client.
	 */
	iter->counts[ref->client_id] -= 1;
	if (0 == iter->counts[ref->client_id]) {
		rc = iter_obj_add(iter, ref->client_id);
		if (rc) {
			/* This client is exausted, -1 means empty */
			iter->counts[ref->client_id] = -1;
		}
	}
	free(ref);
	return obj;
}

static sos_obj_t iter_obj_min(dsos_iter_t iter)
{
	struct ods_rbn *rbn = ods_rbt_min(&iter->obj_tree);
	return iter_obj(iter, rbn);
}

static sos_obj_t iter_obj_max(dsos_iter_t iter)
{
	struct ods_rbn *rbn = ods_rbt_max(&iter->obj_tree);
	return iter_obj(iter, rbn);
}

sos_obj_t dsos_iter_begin(dsos_iter_t iter)
{
	int client_id;
	iter->action = DSOS_ITER_BEGIN;
	for (client_id = 0; client_id < iter->cont->sess->client_count; client_id++) {
		(void)iter_obj_add(iter, client_id);
	}
	iter->action = DSOS_ITER_NEXT;
	return iter_obj_min(iter);
}

sos_obj_t dsos_iter_end(dsos_iter_t iter)
{
	int client_id;
	for (client_id = 0; client_id < iter->cont->sess->client_count; client_id++) {
		(void)iter_obj_add(iter, client_id);
	}
	return iter_obj_max(iter);
}

sos_obj_t dsos_iter_next(dsos_iter_t iter)
{
	return iter_obj_min(iter);
}

sos_obj_t dsos_iter_prev(dsos_iter_t iter)
{
	return iter_obj_max(iter);
}

int dsos_iter_find_glb(dsos_iter_t iter, sos_key_t key)
{
	return ENOSYS;
}

int dsos_iter_find_lub(dsos_iter_t iter, sos_key_t key)
{
	return ENOSYS;
}

int dsos_iter_find(dsos_iter_t iter, sos_key_t key)
{
	return ENOSYS;
}

int dsos_attr_value_min(dsos_container_t cont, sos_attr_t attr)
{
	return ENOSYS;
}

int dsos_attr_value_max(dsos_container_t cont, sos_attr_t attr)
{
	return ENOSYS;
}
/*
 * Query
 */
dsos_query_t dsos_query_create(dsos_container_t cont)
{
	dsos_query_options opts;
	enum clnt_stat rpc_err;
	int i, res;
	dsos_query_create_res create_res;
	dsos_query_t query;

	query = calloc(1, sizeof *query);
	if (!query)
		return NULL;
	query->cont = cont;
	query->handles = calloc(cont->handle_count, sizeof(dsos_query_id));
	if (!query->handles) {
		errno = ENOMEM;
		goto err_0;
	}
	query->counts = calloc(cont->handle_count, sizeof(int));
	if (!query->counts) {
		errno = ENOMEM;
		goto err_1;
	}
	ods_rbt_init(&query->obj_tree, key_comparator, NULL);
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		memset(&create_res, 0, sizeof(create_res));
		memset(&opts, 0, sizeof(opts));
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = query_create_1(cont->handles[i], opts,
					 &create_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr,
				"query_create_1 failed on client %d with RPC"
				" error %d\n",
				i, rpc_err);
			errno = ENETDOWN;
			goto err_2;
		}
		if (create_res.error) {
			fprintf(stderr,
				"query_create_1 failed on client %d with "
				"error %d\n",
				i, create_res.error);
			errno = create_res.error;
			goto err_2;
		}
		query->handles[i] = create_res.dsos_query_create_res_u.query_id;
	}
	query->state = DSOS_QUERY_INIT;
	return query;
err_2:
	for (; i > 0; i--) {
		dsos_client_t client = &cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = query_delete_1(cont->handles[i-1], query->handles[i-1],
					&res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
	}
	free(query->counts);
err_1:
	free(query->handles);
err_0:
	free(query);
	return NULL;
}

/*
 * Remove all objects from the query's object tree
 */
static void reset_query_obj_tree(dsos_query_t query)
{
	sos_obj_t obj;
	dsos_obj_ref_t ref;
	struct ods_rbn *rbn;

	while (NULL != (rbn = ods_rbt_min(&query->obj_tree))) {
		ref = container_of(rbn, struct dsos_obj_ref_s, rbn);
		obj = ref->obj;
		ods_rbt_del(&query->obj_tree, rbn);
		sos_value_put(&ref->key_value);
		sos_obj_put(obj);
		free(ref);
	}
}

void dsos_query_destroy(dsos_query_t query)
{
	int i, res;
	enum clnt_stat rpc_err;

	for (i = 0; i < query->cont->handle_count; i++) {
		dsos_client_t client = &query->cont->sess->clients[i];
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = query_delete_1(query->cont->handles[i], query->handles[i],
					&res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr,
				"query_delete_1 failed on client %d "
				"with RPC error %d\n",
				i, rpc_err);
		}
		if (res) {
			fprintf(stderr,
				"query_delete_1 failed on client %d with error %d\n",
				i, res);
		}
	}
	reset_query_obj_tree(query);
	free(query->handles);
	free(query->counts);
	free(query);
}

int dsos_query_select(dsos_query_t query, const char *clause)
{
	dsos_query_select_res res;
	int client_id;
	enum clnt_stat rpc_err;
	for (client_id = 0; client_id < query->cont->sess->client_count; client_id++) {
		dsos_client_t client = &query->cont->sess->clients[client_id];
		memset(&res, 0, sizeof(res));
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = query_select_1(query->cont->handles[client_id],
					 query->handles[client_id],
					 (char *)clause, &res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr,
				"query_select_1 failed on client %d with "
				"RPC error %d\n",
				client_id, rpc_err);
			return RPC_ERROR(rpc_err);
		}
		if (res.error) {
			fprintf(stderr, "%s: %s\n", __func__,
				res.dsos_query_select_res_u.error_msg);
			return res.error;
		}
		if (query->schema == NULL) {
			query->schema =
				dsos_schema_from_spec(res.dsos_query_select_res_u.select.spec);
			if (!query->schema)
				return DSOS_ERR_SCHEMA;
			query->key_attr = sos_schema_attr_by_id(query->schema,
								res.dsos_query_select_res_u.select.key_attr_id);
			if (!query->key_attr)
				return DSOS_ERR_ATTR;
		}
	}
	query->state = DSOS_QUERY_SELECT;
	return 0;
}

static int query_obj_add(dsos_query_t query, int client_id)
{
	dsos_container_id cont_id = query->cont->handles[client_id];
	dsos_query_id query_id = query->handles[client_id];
	enum clnt_stat rpc_err;
	dsos_query_next_res next_res;
	dsos_obj_entry *obj_e;
	int count = 0;
	dsos_client_t client = &query->cont->sess->clients[client_id];

	memset(&next_res, 0, sizeof(next_res));
	pthread_mutex_lock(&client->rpc_lock);
	rpc_err = query_next_1(cont_id, query_id, &next_res, client->client);
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS)
		return RPC_ERROR(rpc_err);
	if (next_res.error)
		return next_res.error;

	/* Build the objects from the list */
	obj_e = next_res.dsos_query_next_res_u.result.obj_list;
	while (obj_e) {
		dsos_obj_entry *next_obj_e = obj_e->next;

		dsos_obj_ref_t obj_ref = malloc(sizeof *obj_ref);
		assert(obj_ref);
		obj_ref->client_id = client_id;
		sos_obj_t obj = sos_obj_new_with_data(
					query->schema,
					obj_e->value.dsos_obj_value_val,
					obj_e->value.dsos_obj_value_len);
		assert(obj);
		obj_ref->obj = obj;
		sos_value_init(&obj_ref->key_value, obj, query->key_attr);
		ods_rbn_init(&obj_ref->rbn, &obj_ref->key_value);
		ods_rbt_ins(&query->obj_tree, &obj_ref->rbn);
		query->counts[client_id] += 1;
		count += 1;
		free(obj_e->value.dsos_obj_value_val);
		free(obj_e);

		obj_e = next_obj_e;
	}
	if (!count)
		return ENOENT;
	return 0;
}

static sos_obj_t query_obj(dsos_query_t query, struct ods_rbn *rbn)
{
	sos_obj_t obj;
	dsos_obj_ref_t ref;
	int rc;

	if (!rbn)
		return NULL;

	ref = container_of(rbn, struct dsos_obj_ref_s, rbn);
	obj = ref->obj;
	ods_rbt_del(&query->obj_tree, rbn);
	sos_value_put(&ref->key_value);
	/*
	 * If all objects from a given client have been consumed,
	 * replenish the tree with more objects from that client.
	 */
	query->counts[ref->client_id] -= 1;
	if (0 == query->counts[ref->client_id]) {
		rc = query_obj_add(query, ref->client_id);
		if (rc) {
			errno = ENOENT;
			/* This client is exausted, -1 means empty */
			query->counts[ref->client_id] = -1;
		}
	}
	free(ref);
	return obj;
}

static sos_obj_t query_obj_min(dsos_query_t query)
{
	struct ods_rbn *rbn = ods_rbt_min(&query->obj_tree);
	return query_obj(query, rbn);
}

sos_obj_t dsos_query_next(dsos_query_t query)
{
	int client_id, rc;
	switch (query->state) {
	case DSOS_QUERY_NEXT:
		break;
	case DSOS_QUERY_SELECT:
		query->state = DSOS_QUERY_EMPTY;
		for (client_id = 0; client_id < query->cont->sess->client_count; client_id++) {
			rc = query_obj_add(query, client_id);
			if (!rc)
				query->state = DSOS_QUERY_NEXT;
		}
		break;
	case DSOS_QUERY_INIT:
		errno = EINVAL;
		return NULL;
	case DSOS_QUERY_EMPTY:
		errno = ENOENT;
		return NULL;
	}
	return query_obj_min(query);
}

sos_schema_t dsos_query_schema(dsos_query_t query)
{
	return query->schema;
}

sos_attr_t dsos_query_index_attr(dsos_query_t query)
{
	return query->key_attr;
}
