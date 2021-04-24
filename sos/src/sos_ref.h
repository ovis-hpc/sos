#ifndef _SOS_REF_H_
#define _SOS_REF_H_
#include <sys/queue.h>
#include <assert.h>
#include <pthread.h>

#ifdef _SOS_REF_TRACK_
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

static inline char **sos_ref_stack_trace(int *trace_size)
{
	extern size_t backtrace(void *buf, size_t buf_size);
	extern char **backtrace_symbols(void *buf, size_t buf_size);
	void *trace[16];
	char **symbols;
	*trace_size = backtrace(trace, 16);
	symbols = backtrace_symbols(trace, *trace_size);
	return symbols;
}

static inline void sos_ref_stack_dump(FILE *f)
{
	int i, trace_size;
	char **trace = sos_ref_stack_trace(&trace_size);
	for (i = 1; i < trace_size; i++)
		fprintf(f, "\t%s\n", trace[i]);
}

typedef struct sos_ref_inst_s {
	const char *get_func;
	const char *put_func;
	int get_line;
	int put_line;
	const char *name;
	int ref_count;
	LIST_ENTRY(sos_ref_inst_s) entry;
} *sos_ref_inst_t;
#endif

typedef void (*sos_ref_free_fn_t)(void *arg);
typedef struct sos_ref_s {
	int ref_count;		/* for all ref instances */
	sos_ref_free_fn_t free_fn;
	void *free_arg;
#ifdef _SOS_REF_TRACK_
	pthread_mutex_t lock;
	LIST_HEAD(, sos_ref_inst_s) head;
#endif
} *sos_ref_t;

static inline int _sos_ref_put(sos_ref_t r, const char *name, const char *func, int line)
{
	int count;
#ifdef _SOS_REF_TRACK_
	sos_ref_inst_t inst;
	assert(r->ref_count);
	pthread_mutex_lock(&r->lock);
	LIST_FOREACH(inst, &r->head, entry) {
		if (0 == strcmp(inst->name, name)) {
			if (0 == inst->ref_count) {
				fprintf(stderr,
					"name %s func %s line %d put "
					"of zero reference:\n",
					name, func, line);
				sos_ref_stack_dump(stderr);
				assert(0);
			}
			inst->put_func = func;
			inst->put_line = line;
			__sync_sub_and_fetch(&inst->ref_count, 1);
			count = __sync_sub_and_fetch(&r->ref_count, 1);
			goto out;
		}
	}
	fprintf(stderr,
		"name %s ref_count %d func %s line %d put but not taken\n",
		name, r->ref_count, func, line);
	sos_ref_stack_dump(stderr);
	assert(0);
 out:
	if (!count)
		r->free_fn(r->free_arg);
	else
		pthread_mutex_unlock(&r->lock);
#else
	count = __sync_sub_and_fetch(&r->ref_count, 1);
	if (!count)
		r->free_fn(r->free_arg);
#endif
	return count;
}
#define sos_ref_put(_r_, _n_) _sos_ref_put((_r_), (_n_), __func__, __LINE__)

static inline void _sos_ref_get(sos_ref_t r, const char *name, const char *func, int line)
{
#ifdef _SOS_REF_TRACK_
	sos_ref_inst_t inst;
	pthread_mutex_lock(&r->lock);
	if (0 == r->ref_count) {
		fprintf(stderr, "name %s func %s line %d use after free\n",
			name, func, line);
		sos_ref_stack_dump(stderr);
		assert(0);
	}
	LIST_FOREACH(inst, &r->head, entry) {
		if (0 == strcmp(inst->name, name)) {
			__sync_fetch_and_add(&inst->ref_count, 1);
			__sync_fetch_and_add(&r->ref_count, 1);
			inst->get_func = func;
			inst->get_line = line;
			goto out;
		}
	}

	/* No reference with this name exists yet */
	inst = calloc(1, sizeof *inst); assert(inst);
	inst->get_func = func;
	inst->get_line = line;
	inst->name = name;
	inst->ref_count = 1;
	__sync_fetch_and_add(&r->ref_count, 1);
	LIST_INSERT_HEAD(&r->head, inst, entry);
 out:
	pthread_mutex_unlock(&r->lock);
#else
	__sync_fetch_and_add(&r->ref_count, 1);
#endif
}
#define sos_ref_get(_r_, _n_) _sos_ref_get((_r_), (_n_), __func__, __LINE__)

static inline void _sos_ref_init(sos_ref_t r, const char *name,
			     sos_ref_free_fn_t fn, void *arg,
			     const char *func, int line)
{
#ifdef _SOS_REF_TRACK_
	sos_ref_inst_t inst;
	pthread_mutex_init(&r->lock, NULL);
	LIST_INIT(&r->head);
	inst = calloc(1, sizeof *inst); assert(inst);
	inst->get_func = func;
	inst->get_line = line;
	inst->name = name;
	inst->ref_count = 1;
	LIST_INSERT_HEAD(&r->head, inst, entry);
#endif
	r->free_fn = fn;
	r->free_arg = arg;
	r->ref_count = 1;
}
#define sos_ref_init(_r_, _n_, _f_, _a_) _sos_ref_init((_r_), (_n_), (_f_), (_a_), __func__, __LINE__)

/*
 * NOTE: This function is for debugging. `__attribute__((unused))` will
 * suppress the `-Werror=unused-function` for this function.
 */
__attribute__((unused))
static void sos_ref_dump_no_lock(sos_ref_t r, const char *name, FILE *f)
{
#ifdef _SOS_REF_TRACK_
	sos_ref_inst_t inst;
	fprintf(f, "... %s: ref %p free_fn %p free_arg %p ...\n",
		name, r, r->free_fn, r->free_arg);
	fprintf(f,
		"%-16s %-8s %-32s %-32s\n", "Name", "Count", "Get Loc", "Put Loc");
	fprintf(stderr,
		"---------------- -------- -------------------------------- "
		"--------------------------------\n");
	LIST_FOREACH(inst, &r->head, entry) {
		fprintf(f,
			"%-16s %8d %-23s/%8d %-23s/%8d\n",
			inst->name, inst->ref_count, inst->get_func, inst->get_line,
			inst->put_func, inst->put_line);
	}
	fprintf(f, "%16s %8d\n", "Total", r->ref_count);
#endif
}

/*
 * NOTE: This function is for debuggging. `__attribute__((unused))` will
 * suppress the `-Werror=unused-function` for this function.
 */
__attribute__((unused))
static void sos_ref_dump(sos_ref_t r, const char *name, FILE *f)
{
#ifdef _SOS_REF_TRACK_
	pthread_mutex_lock(&r->lock);
	sos_ref_dump_no_lock(r, name, f);
	pthread_mutex_unlock(&r->lock);
#endif
}

__attribute__((unused))
static void sos_ref_assert_count_ge(sos_ref_t r, const char *name, int count)
{
#ifdef _SOS_REF_TRACK_
	sos_ref_inst_t inst;
	LIST_FOREACH(inst, &r->head, entry) {
		if (0 == strcmp(inst->name, name)) {
			assert(inst->ref_count >= count);
			return;
		}
	}
	assert("Reference not present\n");
#endif
}

#endif /* _SOS_REF_H_ */

