#include <stddef.h>
#include <coll/idx.h>

static void get_value_key(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	uint32_t kv;
	uint32_t limit = 100;
	uint64_t v = *(uint64_t *)sos_attr_get(attr, obj);

	kv = 0;
	key->len = 4;
	do {
		if (v < limit)
			break;
		limit += 100;
		kv++;
	} while (1);
	kv = htobe32(kv);
	memcpy(key->value, (unsigned char *)&kv, 4);
}

static void set_key_value(sos_attr_t attr, void *value, obj_key_t key)
{
	uint32_t kv = *(uint32_t *)value;
	kv = htobe32(kv);
	memcpy(key->value, (unsigned char *)&kv, 4);
	key->len = 4;
}

SOS_OBJ_BEGIN(ovis_metric_class, "OvisMetric")
	SOS_OBJ_ATTR_WITH_KEY("tv_sec", SOS_TYPE_UINT32),
	SOS_OBJ_ATTR("tv_usec", SOS_TYPE_UINT32),
	SOS_OBJ_ATTR_WITH_KEY("comp_id", SOS_TYPE_UINT32),
	SOS_OBJ_ATTR_WITH_KEY("value", SOS_TYPE_UINT64),
	SOS_OBJ_ATTR_WITH_KEY("string", SOS_TYPE_STRING),
	SOS_OBJ_ATTR_WITH_KEY("blob", SOS_TYPE_BLOB),
SOS_OBJ_END(6);

idx_t ct_idx;
idx_t c_idx;
struct metric_store_s {
	sos_t sos;
	char *key;
	LIST_ENTRY(metric_store_s) entry;
};
LIST_HEAD(ms_q, metric_store_s) ms_head;

void print_header(struct metric_store_s *m, const char *attr_name)
{
	printf("\n%s by %s\n", m->key, attr_name);
	printf("%11s %16s %8s %12s %12s %12s\n", "Key", "Time", "CompID", "Value", "String", "Blob");
	printf("----------- ---------------- -------- ------------ ------------ ------------\n");
}

void dump_metric_store_fwd(struct metric_store_s *m,
			   int attr_id)
{
	int rc;
	sos_obj_t obj;
	sos_iter_t iter = sos_attr_iter_new(m->sos, attr_id);
	print_header(m, sos_iter_name(iter));
	for (rc = sos_iter_begin(iter); !rc; rc = sos_iter_next(iter)) {
		obj = sos_iter_obj(iter);
		print_obj(m->sos, obj, attr_id);
	}
	sos_iter_free(iter);
}

void dump_metric_store_bkwd(struct metric_store_s *m,
			    int attr_id)
{
	int rc;
	sos_obj_t obj;
	sos_iter_t iter = sos_attr_iter_new(m->sos, attr_id);
	print_header(m, sos_iter_name(iter));
	for (rc = sos_iter_end(iter); !rc; rc = sos_iter_prev(iter)) {
		obj = sos_iter_obj(iter);
		print_obj(m->sos, obj, attr_id);
	}
	sos_iter_free(iter);
}

char tmp_path[PATH_MAX];
int main(int argc, char *argv[])
{
	char *s;
	static char pfx[32];
	static char buf[128];
	static char buf2[128] = {0};
	sos_blob_obj_t blob = (void*)buf2;
	char *str;
	static char c_key[32];
	static char comp_type[32];
	static char metric_name[32];
	struct metric_store_s *m;

	if (argc < 2) {
		printf("usage: ./sos <dir>\n");
		exit(1);
	}
	strcpy(pfx, argv[1]);
	ct_idx = idx_create();
	c_idx = idx_create();

	while ((s = fgets(buf, sizeof(buf), stdin)) != NULL) {
		uint32_t comp_id;
		uint32_t tv_sec;
		uint32_t tv_usec;
		uint64_t value;
		int n;

		sscanf(buf, "%[^,],%[^,],%d,%ld,%d,%d,%s",
		       comp_type, metric_name,
		       &comp_id,
		       &value,
		       &tv_usec,
		       &tv_sec,
		       blob->data);
		blob->len = strlen(blob->data) + 1;

		/*
		 * Add a component type directory if one does not
		 * already exist
		 */
		if (!idx_find(ct_idx, &comp_type, 2)) {
			sprintf(tmp_path, "%s/%s", pfx, comp_type);
			mkdir(tmp_path, 0777);
			idx_add(ct_idx, &comp_type, 2, (void *)1UL);
		}
		sprintf(c_key, "%s:%s", comp_type, metric_name);
		m = idx_find(c_idx, c_key, strlen(c_key));
		if (!m) {
			/*
			 * Open a new SOS for this component-type and
			 * metric combination
			 */
			m = malloc(sizeof *m);
			sprintf(tmp_path, "%s/%s/%s", pfx, comp_type, metric_name);
			m->key = strdup(c_key);
			m->sos = sos_open_sz(tmp_path, O_CREAT | O_RDWR, 0660,
					  &ovis_metric_class, SOS_INITIAL_SIZE);
			if (m->sos) {
				idx_add(c_idx, c_key, strlen(c_key), m);
				LIST_INSERT_HEAD(&ms_head, m, entry);
			} else {
				free(m);
				printf("Could not create SOS database '%s'\n",
				       tmp_path);
				exit(1);
			}
		}
		/* Allocate a new object */
		sos_obj_t obj = sos_obj_new(m->sos);
		if (!obj)
			goto err;

		obj_ref_t objref = ods_obj_ptr_to_ref(m->sos->ods, obj);

		sos_obj_attr_set(m->sos, 0, obj, &tv_sec);
		sos_obj_attr_set(m->sos, 1, obj, &tv_usec);
		sos_obj_attr_set(m->sos, 2, obj, &comp_id);
		sos_obj_attr_set(m->sos, 3, obj, &value);
		sos_obj_attr_set(m->sos, 4, obj, blob->data); /* string */
		/* obj may stale due to possible ods_extend */
		obj = ods_obj_ref_to_ptr(m->sos->ods, objref);
		sos_obj_attr_set(m->sos, 5, obj, blob); /* blob */
		/* obj may stale due to possible ods_extend */
		obj = ods_obj_ref_to_ptr(m->sos->ods, objref);

		/* Add it to the indexes */
		if (sos_obj_add(m->sos, obj))
			goto err;
	}
	/*
	 * Iterate forwards through all the objects we've added and
	 * print them out
	 */
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_fwd(m, 3);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_fwd(m, 0);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_fwd(m, 2);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_fwd(m, 4);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_fwd(m, 5);
	}
	/*
	 * Iterate backwards through all the objects we've added and
	 * print them out
	 */
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_bkwd(m, 3);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_bkwd(m, 0);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_bkwd(m, 2);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_bkwd(m, 4);
	}
	LIST_FOREACH(m, &ms_head, entry) {
		dump_metric_store_bkwd(m, 5);
	}

	LIST_FOREACH(m, &ms_head, entry) {
		sos_close(m->sos, ODS_COMMIT_SYNC);
	}
	return 0;
 err:
	return 1;
}
