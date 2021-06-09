
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
			.type = SOS_TYPE_INT32,
		},
		{
			.name = "Attr_3",
			.type = SOS_TYPE_INT32,
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

typedef struct dsos_iter_s {
	dsos_iter_id iter_id;
	dsos_obj_list_res result;
} *dsos_iter_t;

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
	rpc_err = transaction_begin_1(rqst->cont_id, &rres, client->client);
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
	enum clnt_stat rpc_err = obj_create_1(rqst->request.obj_create.obj_entry, &create_res, client->client);
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
			enum clnt_stat rpc_err = transaction_end_1(cont_id, &rres, client->client);
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
	}

	g_session.next_client = 0;
	return &g_session;
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
		rpc_err = open_1((char *)path, SOS_PERM_RW | SOS_PERM_CREAT, 0660, &open_res, sess->clients[i].client);
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
		if (cont->handles[i])
			rpc_err = close_1(cont->handles[i], &rc, sess->clients[i].client);
	}
	free(cont->path);
	free(cont);
err_0:
	free(cont);
	return NULL;
}

typedef struct dsos_schema_s {
	dsos_container_t cont;	/* The container */
	sos_schema_t schema;	/* The SOS schema */
	int handles[];		/* Array of schema handles from each server */
} *dsos_schema_t;

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
		rpc_err = schema_create_1(cont->handles[i], *spec, &schema_res, cont->sess->clients[i].client);
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
		rpc_err = schema_find_by_name_1(cont->handles[i], (char *)name, &schema_res,
					cont->sess->clients[i].client);
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
		__sync_fetch_and_add(&client->queue_depth, 1);
		client->request_count += 1.0;
		pthread_mutex_lock(&client->queue_lock);
		TAILQ_INSERT_TAIL(&client->queue_q, rqst, link);
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
	client->obj_count += 1.0;
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
	if (rqst && rqst->type == REQ_OBJ_CREATE && rqst->cont_id == cont_id) {
		obj_e->next = rqst->request.obj_create.obj_entry;
		rqst->request.obj_create.obj_entry = obj_e;
	} else {
		rqst = malloc(sizeof *rqst);
		rqst->cont_id = cont->handles[client_id];
		rqst->type = REQ_OBJ_CREATE;
		rqst->request.obj_create.obj_entry = obj_e;
		__sync_fetch_and_add(&client->queue_depth, 1);
		client->request_count += 1.0;
		TAILQ_INSERT_TAIL(&client->queue_q, rqst, link);
	}
	pthread_mutex_unlock(&client->queue_lock);
	pthread_cond_broadcast(&client->queue_cond);
	return;

enomem:
	res->any_err = errno;
	res->res[client_id] = errno;
	return;
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
#if 1
	sos_schema_t schema = sos_schema_from_template(&my_schema_template);
	dschema = dsos_schema_create(cont, schema, &res);
	if (dschema == NULL && res.any_err != EEXIST) {
		printf("Error creating schema\n");
		exit(1);
	}
#endif
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
		sos_obj_attr_by_id_set(obj, 2, i + 2);
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
			if (i < obj_count + 1)
				dsos_transaction_begin(cont, &res);
			clock_gettime(CLOCK_REALTIME, &start);
		}
	}
	sos_obj_put(obj);
	if (i % xcount)
		dsos_transaction_end(cont, &res);

	dsos_session_close(sess);
#if 0
	dsos_iter_res *iter_res = iter_create_1(cont, spec->id, "Attr_1", clnt);
	dsos_iter_id iter_id = iter_res->dsos_iter_res_u.iter;
	dsos_obj_list_res *obj_res = iter_begin_1(cont, iter_id, clnt);

	/* Build the objects from the list */
	while (obj_res->error == 0) {
		dsos_obj_link obj_link = obj_res->dsos_obj_list_res_u.obj_list;
		while (obj_link) {
			int attr_id;
			sos_obj_t obj = sos_obj_new_with_data(schema,
				obj_link->value.dsos_obj_value_val,
				obj_link->value.dsos_obj_value_len);
			obj_link = obj_link->next;
			for (attr_id = 0; attr_id < sos_schema_attr_count(schema); attr_id++) {
				char *s, attr_value[255];
				s = sos_obj_attr_by_id_to_str(obj, attr_id, attr_value, sizeof(attr_value));
				fprintf(stdout, "%s, ", s);
			}
			fprintf(stdout, "\n");
			sos_obj_put(obj);
		}
		xdr_free((xdrproc_t)xdr_dsos_obj_list_res, (char *)obj_res);
		obj_res = iter_next_1(cont, iter_id, clnt);
	}
	xdr_free((xdrproc_t)xdr_dsos_obj_list_res, (char *)obj_res);
	int *close_res = close_1(cont, clnt);
	fprintf(stdout, "close result %d\n", *close_res);
	clnt_destroy (clnt);
#endif
	exit (0);
}
