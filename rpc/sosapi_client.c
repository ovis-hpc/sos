
#define _GNU_SOURCE
#include <sys/queue.h>
#include <sos/sos.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include "sosapi.h"
#include "sosapi_common.h"

static inline double diff_timespec_s(struct timespec start, struct timespec end)
{
        uint64_t nsecs_start, nsecs_end;

        nsecs_start = start.tv_sec * 1000000000 + start.tv_nsec;
        nsecs_end = end.tv_sec * 1000000000 + end.tv_nsec;
        return (double)(nsecs_end - nsecs_start) / 1e9;
}

const char *join_list[] = { "Attr_1", "Attr_2", "Attr_3" };
struct sos_schema_template my_schema_template = {
	.name = "My_Schema",
	.attrs = {
		{
			.name = "Attr_1",
			.type = SOS_TYPE_INT32,
			.indexed = 1,
		},
		{
			.name = "Attr_2",
			.type = SOS_TYPE_INT64,
		},
		{
			.name = "Attr_3",
			.type = SOS_TYPE_DOUBLE,
		},
		{
			.name = "Array_1",
			.type = SOS_TYPE_INT32_ARRAY,
		},
		{
			.name = "Join_1_3",
			.type = SOS_TYPE_JOIN,
			.size = 3,
			.join_list = join_list,
			.indexed = 1,
		},
		{}
	}
};

typedef struct dsos_container_s *dsos_container_t;
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

typedef struct dsos_session_s {
	int client_count;	/* Number of clients */
	int next_client;	/* The next client to use */
	const char **hosts;	/* One for each client */
	struct dsos_client_s *clients;
	enum dsos_session_state_e {
		CONNECTED,
		DISCONNECTED,
		ERROR
	} state;
} *dsos_session_t;

struct dsos_container_s {
	char *path;
	dsos_session_t sess;	/* The session */
	int handle_count;	/* Number of servers */
	int handles[];		/* Array of container handles from each server */
};

typedef struct dsos_schema_s {
	dsos_container_t cont;	/* The container */
	sos_schema_t schema;	/* The SOS schema */
	int handles[];		/* Array of schema handles from each server */
} *dsos_schema_t;

typedef struct dsos_obj_ref_s {
	int client_id;			/* Client this object came from */
	struct sos_value_s key_value;	/* Iterator key value */
	sos_obj_t obj;			/* SOS object */
	struct ods_rbn rbn;		/* obj_tree node */
} *dsos_obj_ref_t;

typedef struct dsos_iter_s {
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
} *dsos_iter_t;

typedef struct dsos_query_s {
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
} *dsos_query_t;

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
	struct transaction_begin_s *begin = &rqst->request.transaction_begin;
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
}

static int handle_obj_create(dsos_client_t client, dsos_client_request_t rqst)
{
	struct create_s *create = &rqst->request.obj_create;
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
	dsos_res_t res;
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
			fprintf(stderr, "Error %d waiting for queue condition variable on client %d\n", rc, client->client_id);
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
	char hostname[256];
	char *s;
	int i;
	int host_count;
	FILE* f = fopen(config_file, "r");
	if (!f)
		return NULL;
	host_count = 0;
	while (s = fgets(hostname, sizeof(hostname), f)) {
		host_count += 1;
	}
	g_session.client_count = host_count;
	g_session.hosts = calloc(host_count, sizeof(void *));
	g_session.clients = calloc(host_count, sizeof(struct dsos_client_s));

	i = 0;
	fseek(f, 0L, SEEK_SET);
	while (s = fgets(hostname, sizeof(hostname), f)) {
		/* Strip the newline if present */
		char *s = strstr(hostname, "\n");
		if (s)
			*s = '\0';
		g_session.hosts[i] = strdup(hostname);
		i += 1;
	}

	for (i = 0; i < host_count; i++) {
		CLIENT *clnt = clnt_create(g_session.hosts[i], SOSDB, SOSVERS, "tcp");
		if (clnt == NULL) {
			fprintf(stderr, "Error creating client %d\n", i);
			exit (1);
		}
		struct timeval timeout = {
			600, 0
		};
		clnt_control(clnt, CLSET_TIMEOUT, (char *)&timeout);
		dsos_client_t client = &g_session.clients[i];
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

	g_session.next_client = 0;
	return &g_session;
}

void dsos_container_close(dsos_container_t cont)
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
	free(cont);
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

dsos_schema_t dsos_schema_by_name(dsos_container_t cont, const char *name, dsos_res_t *res)
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
	int i, rres;
	dsos_client_request_t rqst;
	enum clnt_stat rpc_err;
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

void dsos_obj_create(dsos_container_t cont, dsos_schema_t schema, sos_obj_t obj, dsos_res_t *res)
{
	dsos_create_res *create_res;
	dsos_obj_value value;
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
	enum clnt_stat rpc_err;
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
		int attr_id;

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
	int client_id, rc;
	iter->action = DSOS_ITER_BEGIN;
	for (client_id = 0; client_id < iter->cont->sess->client_count; client_id++) {
		rc = iter_obj_add(iter, client_id);
	}
	iter->action = DSOS_ITER_NEXT;
	return iter_obj_min(iter);
}

sos_obj_t dsos_iter_end(dsos_iter_t iter)
{
	int client_id, rc;
	for (client_id = 0; client_id < iter->cont->sess->client_count; client_id++) {
		rc = iter_obj_add(iter, client_id);
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
		rpc_err = query_create_1(cont->handles[i], opts, &create_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "query_create_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			errno = ENETDOWN;
			goto err_2;
		}
		if (create_res.error) {
			fprintf(stderr, "query_create_1 failed on client %d with error %d\n",
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

int dsos_query_select(dsos_query_t query, const char *clause)
{
	dsos_query_select_res res;
	int client_id;
	enum clnt_stat rpc_err;
	for (client_id = 0; client_id < query->cont->sess->client_count; client_id++) {
		dsos_client_t client = &query->cont->sess->clients[client_id];
		memset(&res, 0, sizeof(res));
		rpc_err = query_select_1(query->cont->handles[client_id], query->handles[client_id],
								(char *)clause, &res, client->client);
		if (query->schema == NULL && res.error == 0) {
			query->schema = dsos_schema_from_spec(res.dsos_query_select_res_u.select.spec);
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
		int attr_id;

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

int main (int argc, char *argv[])
{
	char *config_file, *path;
	dsos_container_t cont;
	int xcount = 10000;
	int obj_count = 1000000;
	dsos_res_t res;

	if (argc < 3) {
		printf ("usage: %s server_host dir\n", argv[0]);
		exit (1);
	}
	config_file = argv[1];
	path = argv[2];
	if (argc >= 4)
		xcount = atoi(argv[3]);
	if (argc == 5)
		obj_count = atoi(argv[4]);

	/* Open the cluster session */
	dsos_session_t sess = dsos_session_open(config_file);
	if (!sess)
		exit(1);

	/* Open the requested container */
	cont = dsos_container_open(sess, path, SOS_PERM_RW | SOS_PERM_CREAT, 0660);
	if (!cont)
		exit(1);
	dsos_schema_t dschema;
	sos_schema_t schema = sos_schema_from_template(&my_schema_template);
	dschema = dsos_schema_create(cont, schema, &res);
	if (dschema == NULL && res.any_err != EEXIST) {
		printf("Error creating schema\n");
		exit(1);
	}
	/* Make certain that we can look it up */
	dschema = dsos_schema_by_name(cont, my_schema_template.name, &res);

	/* Create objects */
	dsos_transaction_begin(cont, &res);
	if (res.any_err) {
		printf("Error starting a transction\n");
		exit(1);
	}
	int i, j;
	uint32_t array[4];
	dsos_obj_value value;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	sos_obj_t obj = sos_obj_malloc(dschema->schema);	/* The local object can be reused when creating remote objects */
	for (i = 1; i < obj_count+1; i++) {
		sos_obj_attr_by_id_set(obj, 0, i);
		sos_obj_attr_by_id_set(obj, 1, i + 1);
		sos_obj_attr_by_id_set(obj, 2, (double)(i + 2));
		for (j = 0; j < 4; j++) {
			array[j] = i + j;
		}
		sos_obj_attr_by_id_set(obj, 3, 4, array);

		dsos_obj_create(cont, dschema, obj, &res);

		if (0 == (i % xcount)) {
			dsos_transaction_end(cont, &res);
			clock_gettime(CLOCK_REALTIME, &end);
			double elapsed = diff_timespec_s(start, end);
			fprintf(stdout, "%10g objects/sec, total %10d, obj/rqst = %g\n", (double)xcount / elapsed, i,
				sess->clients[0].obj_count / sess->clients[0].request_count);
			dsos_transaction_begin(cont, &res);
			clock_gettime(CLOCK_REALTIME, &start);
		}
	}
	sos_obj_put(obj);
	dsos_transaction_end(cont, &res);
	dsos_iter_t iter = dsos_iter_create(cont, dschema, "Attr_1");
	for (obj = dsos_iter_begin(iter); obj; obj = dsos_iter_next(iter)) {
		int attr_id;
		for (attr_id = 0; attr_id < sos_schema_attr_count(schema); attr_id++) {
			char *s, attr_value[255];
			s = sos_obj_attr_by_id_to_str(obj, attr_id, attr_value, sizeof(attr_value));
			fprintf(stdout, "%s, ", s);
		}
		fprintf(stdout, "\n");
		sos_obj_put(obj);
	}

	fprintf(stdout, "------------\n\n");

	dsos_query_t query = dsos_query_create(cont);
	int rc = dsos_query_select(query,
				"select Attr_1, Attr_2, Attr_3 from My_Schema "
				"where ( Attr_1 == 1 ) or ( Attr_2 == 4 ) or (Attr_3 == 5.0)");
	schema = dsos_query_schema(query);
	for (obj = dsos_query_next(query); obj; obj = dsos_query_next(query)) {
		int attr_id;
		for (attr_id = 0; attr_id < sos_schema_attr_count(schema); attr_id++) {
			char *s, attr_value[255];
			s = sos_obj_attr_by_id_to_str(obj, attr_id, attr_value, sizeof(attr_value));
			fprintf(stdout, "%s, ", s);
		}
		fprintf(stdout, "\n");
		sos_obj_put(obj);
	}
	dsos_container_close(cont);
	dsos_session_close(sess);
	return 0;
}
