#define _GNU_SOURCE
#include <sys/queue.h>
#include <sos/sos.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include <semaphore.h>
#include <stdarg.h>
#include "sosapi.h"
#include "sosapi_common.h"
#include "dsos.h"

typedef struct dsos_completion_s {
	pthread_mutex_t lock;
	pthread_cond_t cond;
	int count;		/* when this goes to zero, the condition variable is signaled */
} *dsos_completion_t;

struct dsos_client_s;
typedef struct dsos_client_s *dsos_client_t;
typedef struct dsos_client_request_s {
	LIST_ENTRY(dsos_client_request_s) w_link;	/* submit_wait list link */
	TAILQ_ENTRY(dsos_client_request_s) r_link;	/* request_q link */
	TAILQ_ENTRY(dsos_client_request_s) x_link;	/* x_queue link */
	dsos_client_t client;
	enum clnt_stat rpc_err;
	dsos_completion_t completion;			/* shared between multiple requests */

	enum dsos_client_request_type_e {
		REQ_CONTAINER_OPEN = 1,
		REQ_SCHEMA_QUERY,
		REQ_SCHEMA_BY_NAME,
		REQ_SCHEMA_BY_UUID,
		REQ_PART_QUERY,
		REQ_PART_CREATE,
		REQ_PART_BY_NAME,
		REQ_PART_BY_UUID,
		REQ_PART_STATE_SET,
		REQ_PART_CHOWN,
		REQ_PART_CHMOD,
		REQ_TRANSACTION_BEGIN,
		REQ_TRANSACTION_END,
		REQ_OBJ_CREATE,
		REQ_QUERY_CREATE,
		REQ_QUERY_SELECT,
		REQ_QUERY_NEXT,
		REQ_QUERY_DESTROY
	} kind;

	union {
		struct open_rqst_s {
			dsos_container_t cont;
			char *name;
			sos_perm_t perm;
			int mode;
			dsos_open_res res;
		} open;

		struct schema_query_req_s {
			dsos_container_t cont;
			dsos_schema_query_res res;
			dsos_name_array_t *names;
		} schema_query;

		struct schema_by_name_rqst_s {
			dsos_container_t cont;
			char *name;
			dsos_schema_t schema;
			dsos_schema_res res;
		} schema_by_name;

		struct schema_by_uuid_rqst_s {
			dsos_container_t cont;
			uuid_t uuid;
			dsos_schema_t schema;
			dsos_schema_res res;
		} schema_by_uuid;

		struct part_create_req_s {
			dsos_container_t cont;
			dsos_part_spec spec;
			dsos_part_t part;
			dsos_part_res res;
		} part_create;

		struct part_query_req_s {
			dsos_container_t cont;
			dsos_part_query_res res;
			dsos_name_array_t *names;
		} part_query;

		struct part_by_name_rqst_s {
			dsos_container_t cont;
			char *name;
			dsos_part_t part;
			dsos_part_res res;
		} part_by_name;

		struct part_by_uuid_rqst_s {
			dsos_container_t cont;
			uuid_t uuid;
			dsos_part_t part;
			dsos_part_res res;
		} part_by_uuid;

		struct part_state_set_rqst_s {
			dsos_container_t cont;
			dsos_part_t part;
			sos_part_state_t state;
			int res;
		} part_state_set;

		struct part_chown_rqst_s {
			dsos_container_t cont;
			uid_t uid;
			gid_t gid;
			dsos_part_t part;
			int res;
		} part_chown;

		struct part_chmod_rqst_s {
			dsos_container_t cont;
			int mode;
			dsos_part_t part;
			int res;
		} part_chmod;

		struct transaction_begin_rqst_s {
			dsos_container_t cont;
			dsos_timeval timeout;
			dsos_transaction_res res;
		} transaction_begin;

		struct transaction_end_rqst_s {
			dsos_container_t cont;
			dsos_transaction_res res;
		} transaction_end;

		struct obj_create_rqst_s {
			dsos_container_id cont_id;
			dsos_obj_entry *obj_entry;
			dsos_obj_create_res res;
		} obj_create;

		struct query_create_rqst_s {
			dsos_query_t query;
			dsos_query_options opts;
			dsos_query_create_res res;
		} query_create;

		struct query_select_rqst_s {
			dsos_query_t query;
			dsos_query sql_str;
			dsos_query_select_res res;
		} query_select;

		struct query_next_rqst_s {
			dsos_query_t query;
			dsos_query_options opts;
			dsos_query_next_res res;
		} query_next;

		struct query_destroy_rqst_s {
			dsos_query_t query;
			dsos_query_destroy_res res;
		} query_destroy;
	};
} *dsos_client_request_t;

struct dsos_client_s {
	CLIENT *client;
	pthread_mutex_t rpc_lock;
	int client_id;
	int shutdown;
	pthread_t request_thread;
	int queue_depth;
	double obj_count;
	double request_count;
	/* Transaction timestamps */
	struct timespec x_start;
	struct timespec x_end;
	/* General request queue */
	TAILQ_HEAD(client_request_s, dsos_client_request_s) request_q;
	pthread_mutex_t request_q_lock;
	sem_t request_sem;
	/* Requests queued here between x_start and x_end */
	pthread_mutex_t x_queue_lock;
	pthread_cond_t x_queue_cond;
	TAILQ_HEAD(client_x_queue_q, dsos_client_request_s) x_queue;
};

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
	pthread_mutex_t lock;	/* Mutex across entire session */
};

struct dsos_container_s {
	char *path;
	dsos_session_t sess;	/* The session */
	int error;		/* Last error number */
	char err_msg[256];	/* Last error message from an RPC reply */
	int handle_count;	/* Number of servers */
	int handles[];		/* Array of container handles from each server */
};

struct dsos_schema_s {
	dsos_container_t cont;	/* The container */
	sos_schema_t schema;	/* The SOS schema */
	int handles[];		/* Array of schema handles from each server */
};

struct dsos_part_spec_ref_s {
	dsos_part_spec spec;
	struct dsos_part_spec_ref_s *next;
};

struct dsos_part_s {
	dsos_container_t cont;	/* The container */
	dsos_part_spec *spec;	/* The partition data */
	int handles[];		/* Array of partition handles from each server */
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
	int *counts;			/* Objects in tree from each client */
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
	int *counts;			/* Objects in tree from each client */
	sos_schema_t schema;		/* The schema for objects on this iterator */
	sos_attr_t key_attr;		/* The attribute that is the key */
	pthread_mutex_t obj_tree_lock;
	struct ods_rbt obj_tree;	/* RBT of objects indexed by key_attr */
	char err_msg[256];

};

struct dsos_session_s g_session;
static int g_last_err;
static char g_last_errmsg[1024];

const char *dsos_last_errmsg(void) {
	return g_last_errmsg;
}

const int dsos_last_err(void) {
	return g_last_err;
}

const char *dsos_query_errmsg(dsos_query_t query)
{
	return query->err_msg;
}

static inline void dsos_res_init(dsos_session_t sess, dsos_res_t *res) {
	int i;
	res->count = sess->client_count;
	for (i = 0; i < res->count; i++)
		res->res[i] = 0;
	res->any_err = 0;
}

void submit_request(dsos_client_t client, dsos_client_request_t request)
{
	__sync_fetch_and_add(&request->completion->count, 1);
	pthread_mutex_lock(&client->request_q_lock);
	TAILQ_INSERT_TAIL(&client->request_q, request, r_link);
	pthread_mutex_unlock(&client->request_q_lock);
	sem_post(&client->request_sem);
}

LIST_HEAD(dsos_wait_list_s, dsos_client_request_s);

typedef int (*dsos_request_completion_fn_t)
(dsos_client_t client, dsos_client_request_t request, dsos_res_t *res);

static void format_request_va(dsos_client_request_t request, va_list ap)
{
	switch (request->kind) {
	case REQ_CONTAINER_OPEN:
		memset(&request->open.res, 0, sizeof(request->open.res));
		request->open.cont = va_arg(ap, dsos_container_t);
		request->open.name = va_arg(ap, char *);
		request->open.perm = va_arg(ap, sos_perm_t);
		request->open.mode = va_arg(ap, int);
		break;
	case REQ_SCHEMA_QUERY:
		memset(&request->schema_query, 0, sizeof(request->schema_query));
		request->schema_query.cont = va_arg(ap, dsos_container_t);
		request->schema_query.names = va_arg(ap, dsos_name_array_t *);
		break;
	case REQ_SCHEMA_BY_NAME:
		memset(&request->schema_by_name, 0, sizeof(request->schema_by_name));
		request->schema_by_name.cont = va_arg(ap, dsos_container_t);
		request->schema_by_name.name = va_arg(ap, char *);
		request->schema_by_name.schema = va_arg(ap, dsos_schema_t);
		break;
	case REQ_SCHEMA_BY_UUID:
		memset(&request->part_by_uuid, 0, sizeof(request->part_by_uuid));
		request->part_by_uuid.cont = va_arg(ap, dsos_container_t);
		uuid_copy(request->part_by_uuid.uuid, va_arg(ap, unsigned char *));
		request->schema_by_uuid.schema = va_arg(ap, dsos_schema_t);
		break;
	case REQ_PART_CREATE:
		memset(&request->part_create, 0, sizeof(request->part_create));
		request->part_create.cont = va_arg(ap, dsos_container_t);
		request->part_create.part = va_arg(ap, dsos_part_t);
		request->part_create.spec.name = va_arg(ap, char *);
		request->part_create.spec.path = va_arg(ap, char *);
		request->part_create.spec.desc = va_arg(ap, char *);
		request->part_create.spec.user_id = va_arg(ap, uid_t);
		request->part_create.spec.group_id = va_arg(ap, gid_t);
		request->part_create.spec.perm = va_arg(ap, long);
		break;
	case REQ_PART_QUERY:
		memset(&request->part_query, 0, sizeof(request->part_query));
		request->part_query.cont = va_arg(ap, dsos_container_t);
		request->part_query.names = va_arg(ap, dsos_name_array_t *);
		break;
	case REQ_PART_BY_NAME:
		memset(&request->part_by_name, 0, sizeof(request->part_by_name));
		request->part_by_name.cont = va_arg(ap, dsos_container_t);
		request->part_by_name.name = va_arg(ap, char *);
		request->part_by_name.part = va_arg(ap, dsos_part_t);
		break;
	case REQ_PART_BY_UUID:
		memset(&request->part_by_uuid, 0, sizeof(request->part_by_uuid));
		request->part_by_uuid.cont = va_arg(ap, dsos_container_t);
		uuid_copy(request->part_by_uuid.uuid, va_arg(ap, unsigned char *));
		request->part_by_uuid.part = va_arg(ap, dsos_part_t);
		break;
	case REQ_PART_STATE_SET:
		memset(&request->part_state_set, 0, sizeof(request->part_state_set));
		request->part_state_set.cont = va_arg(ap, dsos_container_t);
		request->part_state_set.part = va_arg(ap, dsos_part_t);
		request->part_state_set.state = va_arg(ap, sos_part_state_t);
		break;
	case REQ_PART_CHOWN:
		memset(&request->part_chown, 0, sizeof(request->part_chown));
		request->part_chown.cont = va_arg(ap, dsos_container_t);
		request->part_chown.part = va_arg(ap, dsos_part_t);
		request->part_chown.uid = va_arg(ap, uid_t);
		request->part_chown.gid = va_arg(ap, gid_t);
		break;
	case REQ_PART_CHMOD:
		memset(&request->part_chmod, 0, sizeof(request->part_chmod));
		request->part_chmod.cont = va_arg(ap, dsos_container_t);
		request->part_chmod.part = va_arg(ap, dsos_part_t);
		request->part_chmod.mode = va_arg(ap, int);
		break;
	case REQ_QUERY_CREATE:
		memset(&request->query_create.res, 0, sizeof(request->query_create.res));
		memset(&request->query_create.opts, 0, sizeof(request->query_create.opts));
		request->query_create.query = va_arg(ap, dsos_query_t);
		break;
	case REQ_QUERY_SELECT:
		memset(&request->query_select.res, 0, sizeof(request->query_select.res));
		request->query_select.query = va_arg(ap, dsos_query_t);
		request->query_select.sql_str = va_arg(ap, char *);
		break;
	case REQ_QUERY_NEXT:
		memset(&request->query_next.res, 0, sizeof(request->query_next.res));
		request->query_next.query = va_arg(ap, dsos_query_t);
		break;
	case REQ_QUERY_DESTROY:
		memset(&request->query_destroy.res, 0, sizeof(request->query_destroy.res));
		request->query_destroy.query = va_arg(ap, dsos_query_t);
		break;
	case REQ_TRANSACTION_BEGIN:
		memset(&request->transaction_begin.res, 0, sizeof(request->transaction_begin.res));
		request->transaction_begin.cont = va_arg(ap, dsos_container_t);
		request->transaction_begin.timeout = va_arg(ap, dsos_timeval);
		break;
	case REQ_TRANSACTION_END:
		memset(&request->transaction_end.res, 0, sizeof(request->transaction_begin.res));
		request->transaction_end.cont = va_arg(ap, dsos_container_t);
		break;
	case REQ_OBJ_CREATE:
		assert(0 == "unsupported");
	}
	return;
}

/**
 * @brief Submit a request to DSOS servers and wait for the result
 *
 * @param sess The session handle
 * @param client_mask A bitmask of servers to which requests will be submitted,
 *             -1 means all servers in the session
 * @param complete_fn The function to call when all servers have responded
 * @param pres A pointer to the dsos_res_t result structure
 * @param kind The type of the request
 * @param ... Request dependent parameters
 * @return 0 Success
 * @return ENOMEM There was insufficient memory to submit the request
 */
int submit_wait(dsos_session_t sess, uint64_t client_mask,
		dsos_request_completion_fn_t complete_fn, dsos_res_t *pres,
		enum dsos_client_request_type_e kind, ...)
{
	int client_id;
	va_list ap;
	struct dsos_wait_list_s wait_list;
	dsos_client_request_t request;

	/* All clients share the same completion */
	dsos_completion_t completion = malloc(sizeof *completion);
	if (!completion)
		return ENOMEM;

	completion->count = 0;
	pthread_mutex_init(&completion->lock, NULL);
	pthread_cond_init(&completion->cond, NULL);

	LIST_INIT(&wait_list);

	for (client_id = 0; client_id < sess->client_count; client_id++) {
		if (0 == ((1L << client_id) & client_mask))
			continue;
		dsos_client_t client = &sess->clients[client_id];

		request = malloc(sizeof *request);
		if (!request)
			goto err_0;

		request->completion = completion;
		request->client = client;
		request->kind = kind;

		va_start(ap, kind);
		format_request_va(request, ap);
		va_end(ap);

		LIST_INSERT_HEAD(&wait_list, request, w_link);
	}
	va_end(ap);

	LIST_FOREACH(request, &wait_list, w_link) {
		__sync_fetch_and_add(&request->completion->count, 1);
		pthread_mutex_lock(&request->client->request_q_lock);
		TAILQ_INSERT_TAIL(&request->client->request_q, request, r_link);
		pthread_mutex_unlock(&request->client->request_q_lock);
		sem_post(&request->client->request_sem);
	}

	pthread_mutex_lock(&completion->lock);
	while (completion->count)
		pthread_cond_wait(&completion->cond, &completion->lock);
	pthread_mutex_unlock(&completion->lock);
	free(completion);

	int rc = 0;
	dsos_res_init(sess, pres);
	while (!LIST_EMPTY(&wait_list)) {
		request = LIST_FIRST(&wait_list);
		LIST_REMOVE(request, w_link);

		complete_fn(request->client, request, pres);

		free(request);
	}
	return pres->any_err;
 err_0:
	free(completion);
	while (!LIST_EMPTY(&wait_list)) {
		request = LIST_FIRST(&wait_list);
		LIST_REMOVE(request, w_link);
		free(request);
	}
	return ENOMEM;
}
#if 0
static int handle_obj_create(dsos_client_t client, dsos_client_request_t rqst)
{
	dsos_create_res create_res = {};

	pthread_mutex_lock(&client->rpc_lock);
	enum clnt_stat rpc_err = obj_create_1(rqst->obj_create.obj_entry, &create_res, client->client);
	pthread_mutex_unlock(&client->rpc_lock);
	if (rpc_err != RPC_SUCCESS) {
		fprintf(stderr, "obj_create_1 failed with RPC error %d\n", rpc_err);
		return rpc_err;
	}
	if (create_res.error)
		fprintf(stderr, "obj_create_1 returned error %d\n", create_res.error);
	dsos_obj_entry *obj_e = rqst->obj_create.obj_entry;
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
void *flush_proc_fn(void *arg)
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
		TAILQ_REMOVE(&client->flush_q, rqst, r_link);
		client->queue_depth -= 1;

		switch (rqst->kind) {
		case REQ_OBJ_CREATE:
			cont_id = rqst->obj_create.obj_entry->cont_id;
			handle_obj_create(client, rqst);
			break;
		case REQ_TRANSACTION_BEGIN:
			cont_id = rqst->transaction_begin.cont_id;
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
		}
	}
	pthread_mutex_unlock(&client->flush_lock);
	goto next;

	return NULL;
}
#endif

static int send_request(dsos_client_t client, dsos_client_request_t rqst)
{
	const char *op_name;
	const char *err_msg;
	switch (rqst->kind) {
	case REQ_OBJ_CREATE:
		memset(&rqst->obj_create.res, 0, sizeof(rqst->obj_create.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			obj_create_1(
				rqst->obj_create.obj_entry,
				&rqst->obj_create.res,
				client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "obj_create_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->obj_create.res.error) {
			g_last_err = rqst->obj_create.res.error;
			err_msg = rqst->obj_create.res.dsos_obj_create_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_TRANSACTION_BEGIN:
		memset(&rqst->transaction_begin.res, 0, sizeof(rqst->transaction_begin.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			transaction_begin_1(
				rqst->transaction_begin.cont->handles[client->client_id],
				rqst->transaction_begin.timeout,
				&rqst->transaction_begin.res,
				client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "transaction_begin_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->transaction_begin.res.error) {
			g_last_err = rqst->transaction_begin.res.error;
			err_msg = rqst->transaction_begin.res.dsos_transaction_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_TRANSACTION_END:
		memset(&rqst->transaction_end.res, 0, sizeof(rqst->transaction_end.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			transaction_end_1(
				rqst->transaction_end.cont->handles[client->client_id],
				&rqst->transaction_end.res,
				client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "transaction_end_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->transaction_end.res.error) {
			g_last_err = rqst->transaction_end.res.error;
			err_msg = rqst->transaction_end.res.dsos_transaction_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_SCHEMA_BY_NAME:
		memset(&rqst->schema_by_name.res, 0, sizeof(rqst->schema_by_name.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			schema_find_by_name_1(rqst->schema_by_name.cont->handles[client->client_id],
					      rqst->schema_by_name.name,
					      &rqst->schema_by_name.res,
					      client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "schema_find_by_name_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->schema_by_name.res.error) {
			g_last_err = rqst->schema_by_name.res.error;
			err_msg =  rqst->schema_by_name.res.dsos_schema_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_SCHEMA_QUERY:
		memset(&rqst->schema_query.res, 0, sizeof(rqst->schema_query.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			schema_query_1(rqst->schema_query.cont->handles[client->client_id],
				       &rqst->schema_query.res,
				       client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "schema_query_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->schema_query.res.error) {
			g_last_err = rqst->schema_query.res.error;
			err_msg =  rqst->schema_query.res.dsos_schema_query_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_SCHEMA_BY_UUID:
		memset(&rqst->schema_by_uuid.res, 0, sizeof(rqst->schema_by_uuid.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			schema_find_by_uuid_1(rqst->schema_by_uuid.schema->cont->handles[client->client_id],
					      (char *)rqst->schema_by_uuid.uuid,
					      &rqst->schema_by_uuid.res,
					      client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "schema_find_by_uuid_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->schema_by_uuid.res.error) {
			g_last_err = rqst->schema_by_uuid.res.error;
			err_msg =  rqst->schema_by_uuid.res.dsos_schema_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_PART_CREATE:
		memset(&rqst->part_create.res, 0, sizeof(rqst->part_create.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			part_create_1(rqst->part_create.cont->handles[client->client_id],
					rqst->part_create.spec,
					&rqst->part_create.res,
					client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "part_create_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->part_create.res.error) {
			g_last_err = rqst->part_create.res.error;
			err_msg =  rqst->part_create.res.dsos_part_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_PART_BY_NAME:
		memset(&rqst->part_by_name.res, 0, sizeof(rqst->part_by_name.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			part_find_by_name_1(rqst->part_by_name.cont->handles[client->client_id],
					      rqst->part_by_name.name,
					      &rqst->part_by_name.res,
					      client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "part_find_by_name_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->part_by_name.res.error) {
			g_last_err = rqst->part_by_name.res.error;
			err_msg =  rqst->part_by_name.res.dsos_part_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_PART_QUERY:
		memset(&rqst->part_query.res, 0, sizeof(rqst->part_query.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			part_query_1(rqst->part_query.cont->handles[client->client_id],
				       &rqst->part_query.res,
				       client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "part_query_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->part_query.res.error) {
			g_last_err = rqst->part_query.res.error;
			err_msg =  rqst->part_query.res.dsos_part_query_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_PART_BY_UUID:
		memset(&rqst->part_by_uuid.res, 0, sizeof(rqst->part_by_uuid.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			part_find_by_uuid_1(rqst->part_by_uuid.part->cont->handles[client->client_id],
					      (char *)rqst->part_by_uuid.uuid,
					      &rqst->part_by_uuid.res,
					      client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "part_find_by_uuid_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->part_by_uuid.res.error) {
			g_last_err = rqst->part_by_uuid.res.error;
			err_msg =  rqst->part_by_uuid.res.dsos_part_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_QUERY_SELECT:
		memset(&rqst->query_select.res, 0, sizeof(rqst->query_select.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			query_select_1(rqst->query_select.query->cont->handles[client->client_id],
				       rqst->query_select.query->handles[client->client_id],
				       rqst->query_select.sql_str,
				       &rqst->query_select.res,
				       client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "query_select_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->query_select.res.error) {
			g_last_err = rqst->query_select.res.error;
			err_msg =  rqst->query_select.res.dsos_query_select_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_QUERY_NEXT:
		memset(&rqst->query_next.res, 0, sizeof(rqst->query_next.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err = query_next_1(rqst->query_next.query->cont->handles[client->client_id],
					     rqst->query_next.query->handles[client->client_id],
					     &rqst->query_next.res,
					     client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "query_next_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->query_next.res.error) {
			g_last_err = rqst->query_next.res.error;
			err_msg = rqst->query_next.res.dsos_query_next_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_CONTAINER_OPEN:
		memset(&rqst->open.res, 0, sizeof(rqst->open.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err = open_1(rqst->open.name, rqst->open.perm, rqst->open.mode,
				       &rqst->open.res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "open_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->open.res.error) {
			g_last_err = rqst->open.res.error;
			err_msg = rqst->open.res.dsos_open_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_QUERY_CREATE:
		memset(&rqst->query_create.res, 0, sizeof(rqst->query_create.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			query_create_1(rqst->query_create.query->cont->handles[client->client_id],
				       rqst->query_create.opts,
				       &rqst->query_create.res,
				       client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "query_create_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->query_create.res.error) {
			g_last_err = rqst->query_create.res.error;
			err_msg = rqst->query_create.res.dsos_query_create_res_u.error_msg;
			goto op_err;
		}
		break;
	case REQ_QUERY_DESTROY:
		memset(&rqst->query_destroy.res, 0, sizeof(rqst->query_destroy.res));
		pthread_mutex_lock(&client->rpc_lock);
		rqst->rpc_err =
			query_destroy_1(rqst->query_destroy.query->cont->handles[client->client_id],
				       rqst->query_destroy.query->handles[client->client_id],
				       &rqst->query_destroy.res,
				       client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		op_name = "query_destroy_1";
		if (rqst->rpc_err != RPC_SUCCESS)
			goto rpc_err;
		if (rqst->query_destroy.res.error) {
			g_last_err = rqst->query_destroy.res.error;
			err_msg = rqst->query_destroy.res.dsos_query_destroy_res_u.error_msg;
			goto op_err;
		}
		break;
	default:
		assert(0 == "Invalid request");
	}
	return 0;
 rpc_err:
	snprintf(g_last_errmsg, sizeof(g_last_errmsg),
		 "%s: failed on client %d with "
		 "RPC error %d\n", op_name,
		 client->client_id, rqst->rpc_err);
	g_last_err = RPC_ERROR(rqst->rpc_err);
	return RPC_ERROR(rqst->rpc_err);
 op_err:
	snprintf(g_last_errmsg, sizeof(g_last_errmsg), "%s: %s", op_name, err_msg);
	return g_last_err;
}

static void *request_proc_fn(void *arg)
{
	dsos_client_t client = arg;
	dsos_client_request_t rqst;
	int rc;
next:
	sem_wait(&client->request_sem);
	pthread_mutex_lock(&client->request_q_lock);
	while (!TAILQ_EMPTY(&client->request_q)) {
		rqst = TAILQ_FIRST(&client->request_q);
		TAILQ_REMOVE(&client->request_q, rqst, r_link);
		pthread_mutex_unlock(&client->request_q_lock);
		/* send the request to the client */
		rc = send_request(client, rqst);
		if (IS_RPC_ERROR(rc))
			printf("Error %d sending RPC. Msg is '%s'\n", rc, dsos_last_errmsg());
		if (0 == __sync_sub_and_fetch(&rqst->completion->count, 1))
			pthread_cond_broadcast(&rqst->completion->cond);
		pthread_mutex_lock(&client->request_q_lock);
	}
	pthread_mutex_unlock(&client->request_q_lock);
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
		pthread_join(client->request_thread, &dontcare);
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
	int rc;
	int host_count;
	FILE* f = fopen(config_file, "r");
	if (!f)
		return NULL;
	host_count = 0;
	while (NULL != (s = fgets(hostname, sizeof(hostname), f))) {
		if (hostname[0] == '#')
			continue;
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
		if (hostname[0] == '#')
			continue;
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
		clnt->cl_auth = authsys_create_default();
#ifdef __not_yet__
		char servername[MAXNETNAMELEN];
		host2netname(servername, session->hosts[i], NULL);
		clnt->cl_auth = authdes_seccreate(servername, 60, session->hosts[i], NULL);
#endif
		clnt_control(clnt, CLSET_TIMEOUT, (char *)&timeout);
		dsos_client_t client = &session->clients[i];
		pthread_mutex_init(&client->rpc_lock, NULL);
		client->client = clnt;
		client->client_id = i;
		client->shutdown = 0;
		client->request_count = 0;
		client->queue_depth = 0;
		client->obj_count = 0;
		TAILQ_INIT(&client->request_q);
		pthread_mutex_init(&client->request_q_lock, NULL);
		sem_init(&client->request_sem, 0, 0);
		TAILQ_INIT(&client->x_queue);
		pthread_mutex_init(&client->x_queue_lock, NULL);
		pthread_cond_init(&client->x_queue_cond, NULL);
		rc = pthread_create(&client->request_thread, NULL, request_proc_fn, client);
		if (rc)
			return NULL;
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "client:%d", (uint8_t)i);
		pthread_setname_np(client->request_thread, thread_name);
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

int dsos_container_error(dsos_container_t cont, const char** err_msg)
{
	if (err_msg)
		*err_msg = cont->err_msg;
	return cont->error;
}

static int
open_complete_fn(dsos_client_t client,
		   dsos_client_request_t request,
		   dsos_res_t *res)
{
	enum dsos_error derr;
	struct dsos_open_res *ores = &request->open.res;
	dsos_container_t cont = request->open.cont;

	if (request->rpc_err) {
		res->any_err = RPC_ERROR(request->rpc_err);
		res->res[request->client->client_id] = res->any_err;
	} else if (request->open.res.error) {
		res->any_err = request->open.res.error;
		res->res[request->client->client_id] = res->any_err;
	} else {
		cont->handles[client->client_id] = request->open.res.dsos_open_res_u.cont;
	}
	return 0;
}

dsos_container_t
dsos_container_open(
		dsos_session_t sess,
		const char *path,
		sos_perm_t perm,
		int mode)
{
	dsos_container_t cont = calloc(1, sizeof *cont + sess->client_count * sizeof(int *));
	if (!cont)
		return NULL;
	pthread_mutex_init(&sess->lock, NULL);
	cont->sess = sess;
	cont->handle_count = sess->client_count;
	cont->path = strdup(path);
	if (!cont->path)
		goto err_0;

	dsos_res_t res;
	int rc = submit_wait(sess, -1, open_complete_fn, &res,
			     REQ_CONTAINER_OPEN, cont, path, perm, mode);
	if (rc) {
		errno = rc;
		return NULL;
	}
	return cont;
 err_0:
	return NULL;
}

static inline dsos_schema_t dsos_schema_alloc(dsos_container_t cont)
{
	dsos_schema_t s;
	size_t size = sizeof(*s) + (cont->handle_count * sizeof(s->handles[0]));
	s = calloc(1, size);
	if (s)
		s->cont = cont;
	return s;
}

static inline void dsos_schema_free(dsos_schema_t schema)
{
	if (!schema)
		return;
	if (schema->schema)
		sos_schema_free(schema->schema);
	free(schema);
}

dsos_schema_t dsos_schema_create(dsos_container_t cont, sos_schema_t schema, dsos_res_t *res)
{
	int i;
	enum clnt_stat rpc_err;
	dsos_schema_t dschema = dsos_schema_alloc(cont);
	dsos_res_init(cont->sess, res);
	dsos_schema_spec *spec = dsos_spec_from_schema(schema);
	if (!spec) {
		res->any_err = errno;
		goto err_0;
	}
	dsos_schema_res schema_res = {};
	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		memset(&schema_res, 0, sizeof(schema_res));
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = schema_create_1(cont->handles[i], *spec, &schema_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			res->any_err = DSOS_ERR_CLIENT;
			break;
		}
		if (schema_res.error) {
			res->any_err = schema_res.error;
			cont->error = schema_res.error;
			strcpy(cont->err_msg, schema_res.dsos_schema_res_u.error_msg);
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
	if (res->any_err)
		goto err_1;
	dsos_spec_free(spec);
	return dschema;
 err_1:
	dsos_spec_free(spec);
 err_0:
	dsos_schema_free(dschema);
	return NULL;
}

static int
schema_by_name_complete_fn(dsos_client_t client,
			   dsos_client_request_t request,
			   dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_schema_res *sres = &request->schema_by_name.res;
	struct schema_by_name_rqst_s *rqst = &request->schema_by_name;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = sres->error;
		if (derr == 0 && rqst->schema->schema == NULL) {
			/*
			 * We only need to instantiate one local
			 * instance of the schema
			 */
			rqst->schema->schema =
				dsos_schema_from_spec(sres->dsos_schema_res_u.spec);
			if (!rqst->schema->schema)
				derr = errno;
		} else {
			/*
			 * Make certain this schema's UUID matches the one
			 * we instantiated for the first schema. Otherwise,
			 * we have different schema with the same name.
			 */
			uuid_t uuid;
			sos_schema_uuid(rqst->schema->schema, uuid);
			if (uuid_compare(sres->dsos_schema_res_u.spec->uuid,
					 uuid)) {
				/* We received a remote schema with a mismatched UUID */
				derr = EEXIST;
				g_last_err = derr;
				snprintf(g_last_errmsg, sizeof(g_last_errmsg),
					 "%s: schema UUID mismatch on client %d\n",
					 __func__, client->client_id);
			}
		}
		request->schema_by_name.schema->handles[client->client_id] =
			sres->dsos_schema_res_u.spec->id;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_schema_res,
		 (char *)&request->schema_by_name.res);
	return 0;
}

dsos_schema_t
dsos_schema_by_name(dsos_container_t cont, const char *name)
{
	dsos_res_t res;
	int rc;
	dsos_schema_t schema = dsos_schema_alloc(cont);
	if (!schema)
		return NULL;
	dsos_res_init(cont->sess, &res);
	rc = submit_wait(cont->sess, -1, schema_by_name_complete_fn, &res,
			 REQ_SCHEMA_BY_NAME, cont, name, schema);
	if (rc)
		goto err_0;
	return schema;
err_0:
	errno = res.any_err;
	dsos_schema_free(schema);
	return NULL;
}

sos_schema_t dsos_schema_local(dsos_schema_t schema)
{
	return schema->schema;
}

static int
schema_by_uuid_complete_fn(dsos_client_t client,
			   dsos_client_request_t request,
			   dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_schema_res *sres =
		&request->schema_by_name.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = sres->error;
		if (derr == 0 && request->schema_by_uuid.schema->schema == NULL) {
			/*
			 * We only need to instantiate one local
			 * instance of the schema
			 */
			request->schema_by_uuid.schema->schema =
				dsos_schema_from_spec(sres->dsos_schema_res_u.spec);
			if (!request->schema_by_uuid.schema->schema)
				derr = errno;
		}
		request->schema_by_uuid.schema->handles[client->client_id] =
			sres->dsos_schema_res_u.spec->id;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&request->schema_by_name.res);
	return 0;
}

dsos_schema_t
dsos_schema_by_uuid(dsos_container_t cont, const uuid_t uuid)
{
	dsos_res_t res;
	int rc;
	dsos_schema_t schema = dsos_schema_alloc(cont);
	if (!schema)
		return NULL;
	dsos_res_init(cont->sess, &res);
	rc = submit_wait(cont->sess, -1, schema_by_uuid_complete_fn, &res,
			 REQ_SCHEMA_BY_UUID, uuid, schema);
	if (rc)
		goto err_0;
	return schema;
err_0:
	free(schema);
	return NULL;
}

sos_schema_t dsos_schema_schema(dsos_schema_t schema)
{
	return schema->schema;
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

void dsos_name_array_free(dsos_name_array_t names)
{
	int i;
	if (!names)
		return;
	for (i = 0; i < names->count; i++)
		free(names->names[i]);
	free(names->names);
	free(names);
}

static int
schema_query_complete_fn(dsos_client_t client,
			 dsos_client_request_t request,
			 dsos_res_t *res)
{
	int i;
	dsos_name_array_t names = NULL;
	enum dsos_error derr = 0;
	struct dsos_schema_query_res *qres =
			&request->schema_query.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = qres->error;
	}
	if (!derr) {
		derr = ENOMEM;
		names = calloc(1, sizeof(*names));
		if (!names)
			goto out;
		names->count = qres->dsos_schema_query_res_u.names.names_len;
		names->names = calloc(names->count, sizeof(char *));
		if (!names->names)
			goto out;
		for (i = 0; i < names->count; i++) {
			names->names[i] =
				strdup(qres->dsos_schema_query_res_u.names.names_val[i]);
			if (!names->names[i])
				goto out;
		}
		*request->schema_query.names = names;
		derr = 0;
	}
out:
	if (derr) {
		if (names) {
			for (i = 0; i < names->count; i++)
				free(names->names[i]);
			free(names->names);
			free(names);
		}
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_schema_query_res, (char *)&request->schema_query.res);
	return 0;
}

dsos_name_array_t dsos_schema_query(dsos_container_t cont)
{
	dsos_res_t res;
	dsos_name_array_t names = NULL;

	dsos_res_init(cont->sess, &res);
	int rc =
		submit_wait(cont->sess, 1, schema_query_complete_fn, &res,
			    REQ_SCHEMA_QUERY, cont, &names);
	if (rc)
		goto err_0;
	return names;
err_0:
	return NULL;
}

int dsos_schema_attr_count(dsos_schema_t schema)
{
	if (!schema->schema)
		return 0;
	return sos_schema_attr_count(schema->schema);
}

void dsos_schema_print(dsos_schema_t schema, FILE *fp)
{
	sos_schema_print(schema->schema, fp);
}

static inline dsos_part_t dsos_part_alloc(dsos_container_t cont)
{
	dsos_part_t p;
	size_t size = sizeof(*p) + (cont->handle_count * sizeof(p->handles[0]));
	p = calloc(1, size);
	if (p)
		p->cont = cont;
	return p;
}

static int
part_create_complete_fn(dsos_client_t client,
			dsos_client_request_t request,
			dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_part_res *pres = &request->part_create.res;
	dsos_part_t part = request->part_create.part;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = pres->error;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	if (!derr) {
		if (!part->spec) {
			part->spec =
				dsos_part_spec_dup(&pres->dsos_part_res_u.spec);
			if (!part->spec) {
				res->res[client->client_id] = ENOMEM;
				if (!res->any_err)
					res->any_err = ENOMEM;
				goto out;
			}
		}
		part->handles[client->client_id] =
			pres->dsos_part_res_u.spec.id;
	}
out:
	if (!request->rpc_err)
		xdr_free((xdrproc_t)xdr_dsos_part_res, (char *)pres);
	return 0;
}

void dsos_part_free(dsos_part_t part)
{
	free(part->spec->path);
	free(part->spec->name);
	free(part->spec->desc);
	free(part->spec);
	free(part);
}

dsos_part_t dsos_part_create(dsos_container_t cont,
				const char *name,
				const char *path,
				const char *desc,
				uid_t uid, gid_t gid, int perm)
{
	dsos_res_t res;
	int rc;
	dsos_part_t part = dsos_part_alloc(cont);
	if (!part)
		return NULL;
	dsos_res_init(cont->sess, &res);
	rc = submit_wait(cont->sess, -1, part_create_complete_fn, &res,
			 REQ_PART_CREATE, cont, part, name, path, desc, uid, gid, perm);
	if (rc)
		goto err_0;
	return part;
err_0:
	free(part);
	return NULL;
}

static int
part_by_name_complete_fn(dsos_client_t client,
			dsos_client_request_t request,
			dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_part_res *sres =
		&request->part_by_name.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = sres->error;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	if (!derr) {
		if (!request->part_by_name.part->spec)
			request->part_by_name.part->spec =
				dsos_part_spec_dup(&request->part_by_name.res.dsos_part_res_u.spec);
		request->part_by_name.part->handles[client->client_id] =
			request->part_by_name.res.dsos_part_res_u.spec.id;
	}
	xdr_free((xdrproc_t)xdr_dsos_part_res, (char *)&request->part_by_name.res);
	return 0;
}

dsos_part_t
dsos_part_by_name(dsos_container_t cont, const char *name)
{
	dsos_res_t res;
	int rc;
	dsos_part_t part = dsos_part_alloc(cont);
	if (!part)
		return NULL;
	dsos_res_init(cont->sess, &res);
	rc = submit_wait(cont->sess, -1, part_by_name_complete_fn, &res,
			 REQ_PART_BY_NAME, cont, name, part);
	if (rc)
		goto err_0;
	return part;
err_0:
	free(part);
	return NULL;
}

static int
part_by_uuid_complete_fn(dsos_client_t client,
			dsos_client_request_t request,
			dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_schema_res *sres =
		&request->schema_by_name.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = sres->error;
		if (derr == 0 && request->schema_by_uuid.schema->schema == NULL) {
			/*
			 * We only need to instantiate one local
			 * instance of the schema.
			 */
			request->schema_by_uuid.schema->schema =
				dsos_schema_from_spec(sres->dsos_schema_res_u.spec);
			if (!request->schema_by_uuid.schema->schema)
				derr = errno;
		}

		request->schema_by_uuid.schema->handles[client->client_id] =
			sres->dsos_schema_res_u.spec->id;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_schema_res, (char *)&request->schema_by_name.res);
	return 0;
}

dsos_part_t
dsos_part_by_uuid(dsos_container_t cont, const uuid_t uuid)
{
	dsos_res_t res;
	int rc;
	dsos_part_t part = dsos_part_alloc(cont);
	if (!part)
		return NULL;
	dsos_res_init(cont->sess, &res);
	rc = submit_wait(cont->sess, -1, part_by_uuid_complete_fn, &res,
			 REQ_PART_BY_UUID, uuid, part);
	if (rc)
		goto err_0;
	return part;
err_0:
	free(part);
	return NULL;
}

static int
part_query_complete_fn(dsos_client_t client,
			dsos_client_request_t request,
			dsos_res_t *res)
{
	int i;
	dsos_name_array_t names = NULL;
	enum dsos_error derr = 0;
	struct dsos_part_query_res *qres =
			&request->part_query.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = qres->error;
	}
	if (!derr) {
		derr = ENOMEM;
		names = calloc(1, sizeof(*names));
		if (!names)
			goto out;
		names->names = calloc(1, sizeof(char *));
		if (!names->names)
			goto out;
		names->count = qres->dsos_part_query_res_u.names.names_len;
		for (i = 0; i < names->count; i++) {
			names->names[i] =
				strdup(qres->dsos_part_query_res_u.names.names_val[i]);
			if (!names->names[i])
				goto out;
		}
		*request->schema_query.names = names;
		derr = 0;
	}
out:
	if (derr) {
		if (names) {
			for (i = 0; i < names->count; i++)
				free(names->names[i]);
			free(names->names);
			free(names);
		}
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_part_query_res, (char *)&request->part_query.res);
	return 0;
}

dsos_name_array_t dsos_part_query(dsos_container_t cont)
{
	dsos_res_t res;
	dsos_name_array_t names = NULL;

	dsos_res_init(cont->sess, &res);
	int rc =
		submit_wait(cont->sess, 1, part_query_complete_fn, &res,
			    REQ_PART_QUERY, cont, &names);
	if (rc)
		goto err_0;
	return names;
err_0:
	return NULL;
}

/* dsos_part_state_set */
static int
part_state_set_complete_fn(dsos_client_t client,
			dsos_client_request_t request,
			dsos_res_t *res)
{
	int derr = 0;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = request->part_state_set.res;
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_int, (char *)&request->part_state_set.res);
	return 0;
}

int dsos_part_state_set(dsos_part_t part, sos_part_state_t state)
{
	dsos_res_t res;
	int rc;
	dsos_res_init(part->cont->sess, &res);
	rc = submit_wait(part->cont->sess, -1, part_state_set_complete_fn, &res,
			 REQ_PART_STATE_SET, part, state);
	if (rc)
		return rc;
	return res.any_err;
}

static int
part_chown_complete_fn(dsos_client_t client,
		       dsos_client_request_t request,
		       dsos_res_t *res)
{
	int derr = 0;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = request->part_chown.res;
	}
	if (!res->any_err)
		res->any_err = derr;

	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_int, (char *)&request->part_chown.res);
	return 0;
}

int dsos_part_chown(dsos_part_t part, uid_t uid, gid_t gid)
{
	dsos_res_t res;
	int rc;
	dsos_res_init(part->cont->sess, &res);
	rc = submit_wait(part->cont->sess, -1, part_chown_complete_fn, &res,
			 REQ_PART_CHOWN, part, uid, gid);
	if (rc)
		return rc;
	return res.any_err;
}

static int
part_chmod_complete_fn(dsos_client_t client,
		       dsos_client_request_t request,
		       dsos_res_t *res)
{
	int derr = 0;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = request->part_chmod.res;
	}
	if (!res->any_err)
		res->any_err = derr;

	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_int, (char *)&request->part_chmod.res);
	return 0;
}

int dsos_part_chmod(dsos_part_t part, int mode)
{
	dsos_res_t res;
	int rc;
	dsos_res_init(part->cont->sess, &res);
	rc = submit_wait(part->cont->sess, -1, part_chmod_complete_fn, &res,
			 REQ_PART_CHMOD, part, mode);
	if (rc)
		return rc;
	return res.any_err;
}

const char *dsos_part_name(dsos_part_t part)
{
	return part->spec->name;
}

const char *dsos_part_desc(dsos_part_t part)
{
	return part->spec->desc;
}

const char *dsos_part_path(dsos_part_t part)
{
	return part->spec->path;
}

void dsos_part_uuid(dsos_part_t part, uuid_t uuid)
{
	uuid_copy(uuid, part->spec->uuid);
}

uid_t dsos_part_uid(dsos_part_t part)
{
	return part->spec->user_id;
}

gid_t dsos_part_gid(dsos_part_t part)
{
	return part->spec->group_id;
}

int dsos_part_perm(dsos_part_t part)
{
	return part->spec->perm;
}

sos_part_state_t dsos_part_state(dsos_part_t part)
{
	return part->spec->state;
}

/*
 * Returns TRUE if the timespec is not zero
 */
static int x_active(struct timespec *ts)
{
	return ts->tv_sec || ts->tv_nsec;
}

static void x_clear(struct timespec *ts)
{
	ts->tv_sec = ts->tv_nsec = 0;
}

/**
 * @brief Begin an object storage transaction
 *
 * A transaction is a read-write boundary for data in the container.
 * When a trnasaction is open, another client attempting to begin
 * a transaction will block and wait. This allows for reader/writer
 * and writer/writer clients on the same machine to maintain a
 * consistent view of the data in the container.
 *
 * In all cases, DSOS maintains data consistency between storage
 * servers.
 *
 * The \c timeout parameter is used to limit time that a client wishes
 * to wait for another client's transaction to end. The caller must check
 * the return code to ensure that a transaction was successffully begun.
 *
 * @param cont The container handle
 * @param timeout Pointer to a timeval structure specifying how long to
 *                wait. If this value is NULL, the function will wait
 *                indefinitely.
 * @return ETIMEDOUT if the timeout expires before the transaction could be opened
 * @return 0 The transaction is open
 */
int dsos_transaction_begin(dsos_container_t cont, struct timeval *timeout)
{
	int client_id;
	int rc = 0;
	dsos_session_t sess = cont->sess;
	struct timespec now;
	(void)clock_gettime(CLOCK_REALTIME, &now);

	pthread_mutex_lock(&sess->lock);
	/* Check if any client is in a transaction and error if so */
	for (client_id = 0; client_id < sess->client_count; client_id++) {
		if (x_active(&sess->clients[client_id].x_start)) {
			rc = EBUSY;
			goto out;
		}
	}
	/* Start the transaction on each client */
	for (client_id = 0; client_id < sess->client_count; client_id++) {
		sess->clients[client_id].x_start = now;
	}
out:
	pthread_mutex_unlock(&sess->lock);
	return rc;
}

static int
transaction_begin_complete_fn(dsos_client_t client,
		   dsos_client_request_t request,
		   dsos_res_t *res)
{
	enum dsos_error derr;
	struct dsos_transaction_res *tres =
		&request->transaction_begin.res;

	if (request->rpc_err) {
		res->any_err = RPC_ERROR(request->rpc_err);
		res->res[request->client->client_id] = res->any_err;
	} else if (request->transaction_begin.res.error) {
		res->any_err = request->transaction_begin.res.error;
		res->res[request->client->client_id] = res->any_err;
	}
	return 0;
}

static int
transaction_end_complete_fn(dsos_client_t client,
		   dsos_client_request_t request,
		   dsos_res_t *res)
{
	enum dsos_error derr;
	struct dsos_transaction_res *tres =
		&request->transaction_end.res;

	if (request->rpc_err) {
		res->any_err = RPC_ERROR(request->rpc_err);
		res->res[request->client->client_id] = res->any_err;
	} else if (request->transaction_end.res.error) {
		res->any_err = request->transaction_end.res.error;
		res->res[request->client->client_id] = res->any_err;
	}
	return 0;
}

/**
 * @brief dsos_transaction_end
 *
 * Submits all queued requests and returns when all requests have completed.
 *
 * @param cont
 * @return int
 */
int dsos_transaction_end(dsos_container_t cont)
{
	dsos_res_t res;
	dsos_session_t sess = cont->sess;
	dsos_client_t client;
	dsos_client_request_t rqst;
	int client_id;
	int rc;
	struct timespec x_end;

	pthread_mutex_lock(&sess->lock);

	/* Tell the server that we're starting a transaction */
	dsos_res_init(sess, &res);
	rc = submit_wait(cont->sess, -1, transaction_begin_complete_fn, &res,
			REQ_TRANSACTION_BEGIN, cont);
	if (rc)
		goto out;
	/* Move all the queued requests to the request queue */
	for (client_id = 0; client_id < sess->client_count; client_id++) {
		client = &sess->clients[client_id];
		pthread_mutex_lock(&client->request_q_lock);
		while (!TAILQ_EMPTY(&client->x_queue)) {
			rqst = TAILQ_FIRST(&client->x_queue);
			TAILQ_REMOVE(&client->x_queue, rqst, x_link);
			TAILQ_INSERT_TAIL(&client->request_q, rqst, r_link);
		}
		pthread_mutex_unlock(&client->request_q_lock);
	}
	/*
	 * Tell the client threads to flush their request queues and wait
	 * for the result
	 */
	dsos_res_init(sess, &res);
	rc = submit_wait(cont->sess, -1,
			transaction_end_complete_fn, &res,
			    REQ_TRANSACTION_END, cont);
	(void)clock_gettime(CLOCK_REALTIME, &x_end);
	for (client_id = 0; client_id < sess->client_count; client_id++) {
		client = &sess->clients[client_id];
		client->x_end = x_end;
		x_clear(&client->x_start);
	}
out:
	pthread_mutex_unlock(&sess->lock);
	return rc;
}

sos_obj_t dsos_obj_new(dsos_schema_t schema)
{
	return sos_obj_malloc(schema->schema);
}

/**
 * @brief Create a DSOS object
 *
 * This creates an instance of the local sos_obj_t on the
 * cluster. The caller must have previously started a
 * transaction to call this function.
 *
 * When the caller calls dsos_transaction_end() all outstanding
 * object creates will be flushed to the storage servers.
 *
 * @param cont	  The container handle
 * @param part	  The partition handle. If NULL, the object will be created
 * 		  in the primary partition
 * @param schema  The object schema
 * @param obj	  The local SOS object
 * @return ENOMEM	There was insufficent local memory to create the object
 * @return 0		The object is queued for creation
 */
int dsos_obj_create(dsos_container_t cont, dsos_part_t part, dsos_schema_t schema, sos_obj_t obj)
{
	dsos_obj_entry *obj_e;
	dsos_client_t client;
	int client_id;
	int rc = 0;
	dsos_container_id cont_id;
	dsos_part_id part_id;

	client_id = __sync_fetch_and_add(&cont->sess->next_client, 1) % cont->sess->client_count;
	client = &cont->sess->clients[client_id];
	cont_id = cont->handles[client_id];
	if (part) {
		part_id = part->handles[client_id];
	} else {
		part_id = -1;
	}
	obj_e = malloc(sizeof *obj_e);
	if (!obj_e)
		goto enomem;

	obj_e->cont_id = cont_id;
	obj_e->part_id = part_id;
	obj_e->schema_id = schema->handles[client_id];
	size_t obj_sz = sos_obj_size(obj);
	obj_e->value.dsos_obj_value_len = obj_sz;
	obj_e->value.dsos_obj_value_val = malloc(obj_sz);
	memcpy(obj_e->value.dsos_obj_value_val, sos_obj_ptr(obj), obj_sz);
	obj_e->next = NULL;

	pthread_mutex_lock(&client->x_queue_lock);
	dsos_client_request_t rqst = TAILQ_LAST(&client->x_queue, client_x_queue_q);
	/*
	 * If there is already an obj create request on the queue, add
	 * this object to the object list for that request
	 */
	client->obj_count += 1.0;
	if (rqst && rqst->kind == REQ_OBJ_CREATE && rqst->obj_create.cont_id == cont_id) {
		obj_e->next = rqst->obj_create.obj_entry;
		rqst->obj_create.obj_entry = obj_e;
	} else {
		dsos_completion_t completion = malloc(sizeof *completion);
		if (!completion)
			goto enomem;
		completion->count = 0;
		pthread_mutex_init(&completion->lock, NULL);
		pthread_cond_init(&completion->cond, NULL);

		rqst = malloc(sizeof *rqst);
		if (!rqst) {
			free(completion);
			goto enomem;
		}
		rqst->completion = completion;
		rqst->obj_create.cont_id = cont->handles[client_id];
		rqst->kind = REQ_OBJ_CREATE;
		rqst->obj_create.obj_entry = obj_e;
		TAILQ_INSERT_TAIL(&client->x_queue, rqst, x_link);
		client->request_count += 1.0;
	}
	pthread_mutex_unlock(&client->x_queue_lock);
enomem:
	return rc;
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
	iter->action = DSOS_ITER_END;
	for (client_id = 0; client_id < iter->cont->sess->client_count; client_id++) {
		(void)iter_obj_add(iter, client_id);
	}
	iter->action = DSOS_ITER_PREV;
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

dsos_iter_stats_t dsos_iter_stats(dsos_iter_t iter)
{
	enum clnt_stat rpc_err;
	int i;
	dsos_container_t cont;
	dsos_iter_stats_res stats_res;
	dsos_iter_stats_t stats;

	memset(&stats, 0, sizeof(stats));

	cont = iter->cont;
	if (!cont)
		goto out;

	for (i = 0; i < cont->handle_count; i++) {
		dsos_client_t client = &cont->sess->clients[i];
		memset(&stats_res, 0, sizeof(stats_res));
		pthread_mutex_lock(&client->rpc_lock);
		rpc_err = iter_stats_1(cont->handles[i], iter->handles[i], &stats_res, client->client);
		pthread_mutex_unlock(&client->rpc_lock);
		if (rpc_err != RPC_SUCCESS) {
			fprintf(stderr, "iter_stats_1 failed on client %d with RPC error %d\n",
				i, rpc_err);
			errno = ENETDOWN;
			goto out;
		}
		if (stats_res.error) {
			fprintf(stderr, "iter_stats_1 failed on client %d with error %d\n",
				i, stats_res.error);
			errno = stats_res.error;
			goto out;
		}
		stats.cardinality += stats_res.dsos_iter_stats_res_u.stats.cardinality;
		stats.duplicates += stats_res.dsos_iter_stats_res_u.stats.duplicates;
		stats.size_bytes += stats_res.dsos_iter_stats_res_u.stats.size_bytes;
	}
 out:
	return stats;
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
static int
query_create_complete_fn(dsos_client_t client,
		   dsos_client_request_t request,
		   dsos_res_t *res)
{
	struct dsos_query_create_res *cres = &request->query_create.res;
	dsos_query_t query = request->query_create.query;
	if (request->rpc_err) {
		if (!res->any_err)
			res->any_err = RPC_ERROR(request->rpc_err);
		res->res[client->client_id] = RPC_ERROR(request->rpc_err);
	} else if (request->query_create.res.error) {
		if (!res->any_err)
			res->any_err = cres->error;
		res->res[client->client_id] = cres->error;
	} else {
		request->query_create.query->handles[client->client_id] =
			request->query_create.res.dsos_query_create_res_u.query_id;
	}
	xdr_free((xdrproc_t)xdr_dsos_query_create_res, (char *)&request->query_create.res);
	return 0;
}

dsos_query_t dsos_query_create(dsos_container_t cont)
{
	dsos_res_t res;
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
	pthread_mutex_init(&query->obj_tree_lock, NULL);
	ods_rbt_init(&query->obj_tree, key_comparator, NULL);

	int rc = submit_wait(cont->sess, -1, query_create_complete_fn, &res,
			 REQ_QUERY_CREATE, query);

	if (!rc)
		query->state = DSOS_QUERY_INIT;
	else
		goto err_2;
	return query;
err_2:
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


static int
query_destroy_complete_fn(dsos_client_t client,
			  dsos_client_request_t request,
			  dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_query_destroy_res *dres = &request->query_destroy.res;

	if (request->rpc_err) {
		res->any_err = RPC_ERROR(request->rpc_err);
		res->res[client->client_id] = res->any_err;
	} else if (dres->error) {
		res->any_err = dres->error;
		res->res[client->client_id] = res->any_err;
	} else {
		res->res[client->client_id] = 0;
	}

	xdr_free((xdrproc_t)xdr_dsos_query_destroy_res, (char *)dres);
	return 0;
}

void dsos_query_destroy(dsos_query_t query)
{
	dsos_res_t res;
	int rc = submit_wait(query->cont->sess, -1, query_destroy_complete_fn, &res,
			     REQ_QUERY_DESTROY, query);
	reset_query_obj_tree(query);
	sos_schema_free(query->schema);
	free(query->handles);
	free(query->counts);
	free(query);
}

static int
query_select_complete_fn(dsos_client_t client,
		   dsos_client_request_t request,
		   dsos_res_t *res)
{
	enum dsos_error derr = 0;
	struct dsos_query_select_res *sres = &request->query_select.res;

	if (request->rpc_err) {
		derr = RPC_ERROR(request->rpc_err);
	} else {
		derr = sres->error;
		if (derr == 0 && request->query_select.query->schema == NULL) {
			request->query_select.query->schema =
				dsos_schema_from_spec(sres->dsos_query_select_res_u.select.spec);
			if (!request->query_select.query->schema)
				derr = DSOS_ERR_SCHEMA;
			request->query_select.query->key_attr =
				sos_schema_attr_by_id(request->query_select.query->schema,
						      sres->dsos_query_select_res_u.select.key_attr_id);
			if (!request->query_select.query->key_attr)
				derr = DSOS_ERR_ATTR;
		}
		if (derr) {
			strcpy(request->query_select.query->err_msg, sres->dsos_query_select_res_u.error_msg);
		}
	}
	if (!res->any_err)
		res->any_err = derr;
	res->res[client->client_id] = derr;
	xdr_free((xdrproc_t)xdr_dsos_query_select_res, (char *)&request->query_select.res);
	return 0;
}

int dsos_query_select(dsos_query_t query, const char *clause)
{
	dsos_res_t res;
	int rc = submit_wait(query->cont->sess, -1, query_select_complete_fn, &res,
			     REQ_QUERY_SELECT, query, clause);
	if (rc)
		return rc;
	if (res.any_err == 0)
		query->state = DSOS_QUERY_SELECT;
	return res.any_err;
}


static int
query_next_complete_fn(dsos_client_t client,
		       dsos_client_request_t request,
		       dsos_res_t *res)
{
	dsos_obj_entry *obj_e;

	/* Build the objects from the list */
	obj_e = request->query_next.res.dsos_query_next_res_u.result.obj_list;
	while (obj_e) {
		dsos_obj_entry *next_obj_e = obj_e->next;
		dsos_obj_ref_t obj_ref = malloc(sizeof *obj_ref);
		assert(obj_ref);
		obj_ref->client_id = request->client->client_id;
		sos_obj_t obj = sos_obj_new_with_data(
						      request->query_next.query->schema,
						      obj_e->value.dsos_obj_value_val,
						      obj_e->value.dsos_obj_value_len);
		assert(obj);
		obj_ref->obj = obj;
		sos_value_init(&obj_ref->key_value, obj, request->query_next.query->key_attr);
		ods_rbn_init(&obj_ref->rbn, &obj_ref->key_value);
		pthread_mutex_lock(&request->query_next.query->obj_tree_lock);
		ods_rbt_ins(&request->query_next.query->obj_tree, &obj_ref->rbn);
		pthread_mutex_unlock(&request->query_next.query->obj_tree_lock);
		request->query_next.query->counts[obj_ref->client_id] += 1;
		obj_e = next_obj_e;
	}
	xdr_free((xdrproc_t)xdr_dsos_query_next_res, (char *)&request->query_next.res);
	return 0;
}

static int query_obj_add(dsos_query_t query, uint64_t client_mask)
{
	dsos_res_t res;
	int rc = submit_wait(query->cont->sess, client_mask, query_next_complete_fn, &res,
			     REQ_QUERY_NEXT, query);
	return res.any_err;
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
		rc = query_obj_add(query, 1L << ref->client_id);
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
		rc = query_obj_add(query, -1);
		if (!rc)
			query->state = DSOS_QUERY_NEXT;
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

/**
 * @brief Return the temporary schema for the \c query
 *
 * @param query The query handle
 * @return sos_schema_t The temporary SOS schema
 */
sos_schema_t dsos_query_schema(dsos_query_t query)
{
	return query->schema;
}

sos_attr_t dsos_query_index_attr(dsos_query_t query)
{
	return query->key_attr;
}
