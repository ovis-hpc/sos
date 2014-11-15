#if 0

static const char *sos_type_to_key_str(enum sos_type_e t)
{
	static const char *key_map[] = {
		"INT32",
		"INT64",
		"UINT32",
		"UINT64",
		"DOUBLE",
		"STRING",
		"BLOB",
	};
	return key_map[t];
}

void sos_commit(sos_t sos, int flags)
{
	int attr_id;

	ods_commit(sos->ods, flags);
	for (attr_id = 0; attr_id < sos->classp->count; attr_id++)
		if (sos->classp->attrs[attr_id].oidx)
			obj_idx_commit(sos->classp->attrs[attr_id].oidx,
				       flags);
}

void sos_close(sos_t sos, int flags)
{
	int attr_id;

	sos_commit(sos, flags);
	ods_close(sos->ods, flags);
	for (attr_id = 0; attr_id < sos->classp->count; attr_id++) {
		obj_idx_close(sos->classp->attrs[attr_id].oidx, flags);
	}
	if (sos->path)
		free(sos->path);
	free(sos);
}

static sos_attr_size_fn_t attr_size_fns[] = {
	[SOS_TYPE_INT32] = SOS_TYPE_INT32__attr_size_fn,
	[SOS_TYPE_INT64] = SOS_TYPE_INT64__attr_size_fn,
	[SOS_TYPE_UINT32] = SOS_TYPE_UINT32__attr_size_fn,
	[SOS_TYPE_UINT64] = SOS_TYPE_UINT64__attr_size_fn,
	[SOS_TYPE_DOUBLE] = SOS_TYPE_DOUBLE__attr_size_fn,
	[SOS_TYPE_STRING] = SOS_TYPE_STRING__attr_size_fn,
	[SOS_TYPE_BLOB] = SOS_TYPE_BLOB__attr_size_fn
};

static sos_get_key_fn_t get_key_fns[] = {
	[SOS_TYPE_INT32] = SOS_TYPE_INT32__get_key_fn,
	[SOS_TYPE_INT64] = SOS_TYPE_INT64__get_key_fn,
	[SOS_TYPE_UINT32] = SOS_TYPE_UINT32__get_key_fn,
	[SOS_TYPE_UINT64] = SOS_TYPE_UINT64__get_key_fn,
	[SOS_TYPE_DOUBLE] = SOS_TYPE_DOUBLE__get_key_fn,
	[SOS_TYPE_STRING] = SOS_TYPE_STRING__get_key_fn,
	[SOS_TYPE_BLOB] = SOS_TYPE_BLOB__get_key_fn
};

static sos_set_key_fn_t set_key_fns[] = {
	[SOS_TYPE_INT32] = SOS_TYPE_INT32__set_key_fn,
	[SOS_TYPE_INT64] = SOS_TYPE_INT64__set_key_fn,
	[SOS_TYPE_UINT32] = SOS_TYPE_UINT32__set_key_fn,
	[SOS_TYPE_UINT64] = SOS_TYPE_UINT64__set_key_fn,
	[SOS_TYPE_DOUBLE] = SOS_TYPE_DOUBLE__set_key_fn,
	[SOS_TYPE_STRING] = SOS_TYPE_STRING__set_key_fn,
	[SOS_TYPE_BLOB] = SOS_TYPE_BLOB__set_key_fn
};

static sos_set_fn_t set_fns[] = {
	[SOS_TYPE_INT32] = SOS_TYPE_INT32__set_fn,
	[SOS_TYPE_INT64] = SOS_TYPE_INT64__set_fn,
	[SOS_TYPE_UINT32] = SOS_TYPE_UINT32__set_fn,
	[SOS_TYPE_UINT64] = SOS_TYPE_UINT64__set_fn,
	[SOS_TYPE_DOUBLE] = SOS_TYPE_DOUBLE__set_fn,
	[SOS_TYPE_STRING] = SOS_TYPE_STRING__set_fn,
	[SOS_TYPE_BLOB] = SOS_TYPE_BLOB__set_fn
};

static sos_get_fn_t get_fns[] = {
	[SOS_TYPE_INT32] = SOS_TYPE_INT32__get_fn,
	[SOS_TYPE_INT64] = SOS_TYPE_INT64__get_fn,
	[SOS_TYPE_UINT32] = SOS_TYPE_UINT32__get_fn,
	[SOS_TYPE_UINT64] = SOS_TYPE_UINT64__get_fn,
	[SOS_TYPE_DOUBLE] = SOS_TYPE_DOUBLE__get_fn,
	[SOS_TYPE_STRING] = SOS_TYPE_STRING__get_fn,
	[SOS_TYPE_BLOB] = SOS_TYPE_BLOB__get_fn
};

int sos_class_cmp(sos_class_t c0, sos_class_t c1)
{
	int i, rc;
	rc = strcmp(c0->name, c1->name);
	if (rc)
		return rc;
	rc = c0->count - c1->count;
	if (rc)
		return rc;
	for (i = 0; i < c0->count; i++) {
		rc = strcmp(c0->attrs[i].name, c1->attrs[i].name);
		if (rc)
			return rc;
		rc = c0->attrs[i].type - c1->attrs[i].type;
		if (rc)
			return rc;
	}
	return 0;
}

static sos_class_t init_classp(sos_meta_t meta)
{
	int attr_id;
	sos_class_t classp =
		calloc(1, sizeof(*classp) + (meta->attr_cnt * sizeof(struct sos_attr_s)));
	if (!classp)
		goto out;
	classp->name = strdup(meta->classname);
	classp->count= meta->attr_cnt;
	for (attr_id = 0; attr_id < meta->attr_cnt; attr_id++) {
		enum sos_type_e at;
		classp->attrs[attr_id].name = strdup(meta->attrs[attr_id].name);
		classp->attrs[attr_id].type = meta->attrs[attr_id].type;
		classp->attrs[attr_id].has_idx = meta->attrs[attr_id].has_idx;
		at = classp->attrs[attr_id].type;
		if (type_is_builtin(at)) {
			classp->attrs[attr_id].attr_size_fn = attr_size_fns[at];
			classp->attrs[attr_id].get_key_fn = get_key_fns[at];
			classp->attrs[attr_id].set_key_fn = set_key_fns[at];
			classp->attrs[attr_id].set_fn = set_fns[at];
			classp->attrs[attr_id].get_fn = get_fns[at];
		}
	}

 out:
	return classp;
}

/*
 * Create/open the indexes as required from the meta data
 */
static sos_class_t init_sos(sos_t sos, int o_flag, int o_mode,
		    sos_meta_t meta, sos_class_t classp)
{
	int attr_id;
	if (!classp) {
		classp = init_classp(meta);
		if (!classp)
			goto err;
	}
	for (attr_id = 0; attr_id < meta->attr_cnt; attr_id++) {
		obj_idx_t oidx;

		classp->attrs[attr_id].id = attr_id;
		classp->attrs[attr_id].sos = sos;
		classp->attrs[attr_id].data = meta->attrs[attr_id].data;

		if (!classp->attrs[attr_id].has_idx) {
			classp->attrs[attr_id].oidx = NULL;
			continue;
		}

		oidx = init_idx(sos, o_flag, o_mode, &classp->attrs[attr_id]);
		if (!oidx)
			goto err;

		classp->attrs[attr_id].oidx = oidx;
	}
	return classp;
 err:
	return NULL;
}

static void free_sos(sos_t sos)
{
	return;
}

sos_class_t dup_class(sos_class_t classp)
{
	size_t sz = sizeof(*classp) +
		(classp->count * sizeof(struct sos_attr_s));
	sos_class_t dup = malloc(sz);
	if (dup)
		memcpy(dup, classp, sz);
	return dup;
}

sos_t sos_open_sz(const char *path, int o_flag, ...)
{
	char tmp_path[PATH_MAX];
	va_list argp;
	int o_mode;
	size_t init_size = 0;
	sos_meta_t meta;
	sos_class_t classp;
	struct stat _stat = {0};
	struct sos_s *sos;

	sos = calloc(1, sizeof(*sos));
	if (!sos)
		goto out;

	sos->path = strdup(path);

	if (o_flag & O_CREAT) {
		va_start(argp, o_flag);
		o_mode = va_arg(argp, int);
		classp = va_arg(argp, sos_class_t);
		init_size = va_arg(argp, size_t);
	} else {
		o_mode = 0;
		classp = NULL;
	}

	if (classp) {
		/* Duplicate the class because we update it with state later */
		classp = dup_class(classp);
		if (!classp) {
			errno = ENOMEM;
			return NULL;
		}
	}

	sprintf(tmp_path, "%s_sos", sos->path);
	sos->ods = ods_open_sz(tmp_path, o_flag, o_mode, init_size);
	if (!sos->ods)
		goto err;

	meta = ods_get_user_data(sos->ods, &sos->meta_sz);
	if (memcmp(meta->signature, SOS_SIGNATURE, sizeof(SOS_SIGNATURE))) {
		/*
		 * You can't create a new repository without a class
		 * definition.
		 */
		if (!classp)
			goto err;

		meta = make_meta(sos, meta, classp);
		if (!meta)
			goto err;
	}
	sos->meta = meta;
	sprintf(tmp_path, "%s_sos.OBJ", sos->path);
	stat(tmp_path, &_stat);
	sos->classp = init_sos(sos, o_flag, _stat.st_mode | o_mode, meta, classp);
	if (!sos->classp)
		goto err;
 out:
	return sos;

 err:
	free_sos(sos);
	return NULL;
}

int sos_extend(sos_t sos, size_t sz)
{
	int rc = ods_extend(sos->ods, sos_meta(sos)->ods_extend_sz);
	if (rc) {
		perror("ods_extend");
		return rc;
	}
	sos->meta = ods_get_user_data(sos->ods, &sos->meta_sz);
	return 0;
}

void sos_obj_delete(sos_t sos, sos_obj_t obj)
{
	int i;
	obj_ref_t ref;
	void *ptr;
	for (i = 0; i < sos->classp->count; i++) {
		sos_attr_t attr = sos_value_by_id(sos, i);
		switch (attr->type) {
		case SOS_TYPE_BLOB:
		case SOS_TYPE_STRING:
			ref = *(obj_ref_t *)&obj->data[attr->data];
			ptr = ods_obj_ref_to_ptr(attr->sos->ods, ref);
			ods_free(attr->sos->ods, ptr);
			break;
		default:
			/* do nothing */
			break;
		}
	}
	ods_free(sos->ods, obj);
}
size_t SOS_TYPE_INT32__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	return sizeof(int32_t);
}

size_t SOS_TYPE_UINT32__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	return sizeof(uint32_t);
}

size_t SOS_TYPE_INT64__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	return sizeof(int64_t);
}

size_t SOS_TYPE_UINT64__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	return sizeof(uint64_t);
}

size_t SOS_TYPE_DOUBLE__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	return sizeof(double);
}

size_t SOS_TYPE_STRING__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	obj_ref_t ref = *(obj_ref_t *)&obj->data[attr->data];
	sos_obj_t strobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	char *str = (char*)strobj->data;
	return strlen(str) + 1;
}

size_t SOS_TYPE_BLOB__attr_size_fn(sos_attr_t attr, sos_obj_t obj)
{
	obj_ref_t ref = *(obj_ref_t *)&obj->data[attr->data];
	sos_obj_t blobobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	sos_blob_obj_t blob = (typeof(blob))blobobj->data;
	return sizeof(*blob) + blob->len;
}

void SOS_TYPE_INT32__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_key_set(key, &obj->data[attr->data], sizeof(int32_t));
}

void SOS_TYPE_UINT32__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_key_set(key, &obj->data[attr->data], sizeof(uint32_t));
}

void SOS_TYPE_INT64__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_key_set(key, &obj->data[attr->data], sizeof(int64_t));
}

void SOS_TYPE_UINT64__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_key_set(key, &obj->data[attr->data], sizeof(uint64_t));
}

void SOS_TYPE_DOUBLE__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_key_set(key, &obj->data[attr->data], sizeof(double));
}

void SOS_TYPE_STRING__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_ref_t ref = *(obj_ref_t *)&obj->data[attr->data];
	sos_obj_t strobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	char *str = (char*)strobj->data;
	obj_key_set(key, str, strlen(str)+1);
}

void SOS_TYPE_BLOB__get_key_fn(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	obj_ref_t ref = *(obj_ref_t *)&obj->data[attr->data];
	sos_obj_t blobobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	sos_blob_obj_t blob = (typeof(blob))blobobj->data;
	obj_key_set(key, blob, sizeof(*blob) + blob->len);
}

void SOS_TYPE_INT32__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, sizeof(int32_t));
}

void SOS_TYPE_UINT32__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, sizeof(uint32_t));
}

void SOS_TYPE_INT64__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, sizeof(int64_t));
}

void SOS_TYPE_UINT64__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, sizeof(uint64_t));
}

void SOS_TYPE_DOUBLE__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, sizeof(double));
}

void SOS_TYPE_STRING__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	obj_key_set(key, value, strlen(value)+1);
}

void SOS_TYPE_BLOB__set_key_fn(sos_attr_t attr, void *value, obj_key_t key)
{
	sos_blob_obj_t blob = value;
	obj_key_set(key, blob, sizeof(*blob) +  blob->len);
}

void SOS_TYPE_INT32__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	*(int32_t *)(&obj->data[attr->data]) = *(int32_t *)value;
}

void SOS_TYPE_UINT32__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	*(uint32_t *)(&obj->data[attr->data]) = *(uint32_t *)value;
}

void SOS_TYPE_INT64__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	*(int64_t *)(&obj->data[attr->data]) = *(int64_t *)value;
}

void SOS_TYPE_UINT64__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	*(uint64_t *)(&obj->data[attr->data]) = *(uint64_t *)value;
}

void SOS_TYPE_DOUBLE__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	*(double *)(&obj->data[attr->data]) = *(double *)value;
}

void SOS_TYPE_STRING__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	obj_ref_t strref = *(obj_ref_t *)&obj->data[attr->data];
	obj_ref_t objref = ods_obj_ptr_to_ref(attr->sos->ods, obj);
	sos_obj_t dst = ods_obj_ref_to_ptr(attr->sos->ods, strref);
	char *src = (char *)value;
	size_t src_len = strlen(src) + 1;
	size_t newstr_sz = src_len + sizeof(struct sos_obj_s);
	if (dst) {
		/* If the memory containing the current value is big enough, use it */
		if (ods_obj_size(attr->sos->ods, dst) >= newstr_sz) {
			strcpy((char*)dst->data, src);
			return;
		} else
			ods_free(attr->sos->ods, dst);
	}
	dst = ods_obj_alloc(attr->sos->ods, newstr_sz);
	if (!dst) {
		if (ods_extend(attr->sos->ods, (newstr_sz | (SOS_ODS_EXTEND_SZ - 1))+1))
			assert(NULL == "ods extend failure.");
		/* ods extended, obj and meta are now stale: update it */
		attr->sos->meta = ods_get_user_data(attr->sos->ods, &attr->sos->meta_sz);
		obj = ods_obj_ref_to_ptr(attr->sos->ods, objref);
		dst = ods_alloc(attr->sos->ods, src_len);
		if (!dst)
			assert(NULL == "ods allocation failure");
		dst->type = SOS_OBJ_TYPE_ATTR;
	}

	strcpy((char*)dst->data, src);
	strref = ods_obj_ptr_to_ref(attr->sos->ods, dst);
	*(obj_ref_t *)&obj->data[attr->data] = strref;
}

void SOS_TYPE_BLOB__set_fn(sos_attr_t attr, sos_obj_t obj, void *value)
{
	ods_t ods = attr->sos->ods;
	obj_ref_t objref = ods_obj_ptr_to_ref(ods, obj);
	obj_ref_t bref = *(obj_ref_t *)&obj->data[attr->data];
	sos_obj_t blob = ods_obj_ref_to_ptr(ods, bref);
	sos_blob_obj_t arg = (typeof(arg))value;
	size_t alloc_len = sizeof(struct sos_obj_s)
				+ sizeof(struct sos_blob_obj_s) + arg->len;

	if (blob && ods_obj_size(ods, blob) < alloc_len) {
		/* Cannot reuse space, free it and reset blob */
		ods_free(ods, blob);
		blob = NULL;
		*(obj_ref_t *)&obj->data[attr->data] = 0;
	}

	if (!blob) {
		/* blob not allocated --> allocate it */
		blob = ods_alloc(ods, alloc_len);
		if (!blob) {
			if (ods_extend(ods, (alloc_len | (SOS_ODS_EXTEND_SZ - 1))+1))
				assert(NULL == "ods extend failure.");
			/* ods extended, obj is now stale: update it */
			attr->sos->meta = ods_get_user_data(attr->sos->ods, &attr->sos->meta_sz);
			obj = ods_obj_ref_to_ptr(attr->sos->ods, objref);
			blob = ods_alloc(ods, alloc_len);
			if (!blob)
				assert(NULL == "ods allocation failure");
		}
		bref = ods_obj_ptr_to_ref(ods, blob);
		blob->type = SOS_OBJ_TYPE_ATTR;
	}

	memcpy(blob->data, arg, sizeof(struct sos_blob_obj_s) + arg->len);
	*(obj_ref_t *)&obj->data[attr->data] = bref;
}

void *SOS_TYPE_INT32__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}

void *SOS_TYPE_UINT32__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}

void *SOS_TYPE_INT64__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}

void *SOS_TYPE_UINT64__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}

void *SOS_TYPE_DOUBLE__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}

void *SOS_TYPE_STRING__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	obj_ref_t ref = *(obj_ref_t*)&obj->data[attr->data];
	sos_obj_t strobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	return strobj->data;
}

void *SOS_TYPE_BLOB__get_fn(sos_attr_t attr, sos_obj_t obj)
{
	obj_ref_t ref = *(obj_ref_t*)&obj->data[attr->data];
	sos_obj_t blobobj = ods_obj_ref_to_ptr(attr->sos->ods, ref);
	return blobobj->data;
}

void sos_value_key(sos_t sos, int attr_id, sos_obj_t obj, obj_key_t key)
{
	sos_attr_t attr = sos_value_by_id(sos, attr_id);
	if (attr)
		sos_attr_key(attr, obj, key);
}

void sos_attr_key(sos_attr_t attr, sos_obj_t obj, obj_key_t key)
{
	attr->get_key_fn(attr, obj, key);
}

int sos_attr_key_cmp(sos_attr_t attr, obj_key_t a, obj_key_t b)
{
	return obj_key_cmp(attr->oidx, a, b);
}

static int __remove_key(sos_t sos, sos_obj_t obj, sos_iter_t iter)
{
	int attr_id;
	size_t attr_sz;
	size_t key_sz = 1024;
	if (!obj)
		obj = sos_iter_obj(iter);
	obj_ref_t obj_ref = ods_obj_ptr_to_ref(sos->ods, obj);
	obj_iter_t oiter;
	obj_ref_t oref;
	int rc;

	/* Delete key at iterator position */
	if (iter) {
		obj_iter_key_del(iter->iter);
	}

	/* Delete the other keys related to the object */
	obj_key_t key = obj_key_new(key_sz);
	for (attr_id = 0; attr_id < sos_meta(sos)->attr_cnt; attr_id++) {
		if (iter && attr_id == iter->attr->id)
			continue;
		sos_attr_t attr = sos_value_by_id(sos, attr_id);
		if (!sos_attr_has_index(attr))
			continue;

		attr_sz = sos_value_size(sos, attr_id, obj);
		if (attr_sz > key_sz) {
			obj_key_delete(key);
			key_sz = attr_sz;
			key = obj_key_new(key_sz);
		}
		sos_attr_key(attr, obj, key);
		oiter = obj_iter_new(attr->oidx);
		assert(oiter);
		rc = obj_iter_find_glb(oiter, key);
		assert(rc == 0);
		oref = obj_iter_ref(oiter);
		while (oref != obj_ref) {
			rc = obj_iter_next(oiter);
			assert(rc == 0);
			oref = obj_iter_ref(oiter);
		}
		obj_iter_key_del(oiter);
		obj_iter_delete(oiter);
	}
	obj_key_delete(key);

	return 0;
}

int sos_obj_remove(sos_t sos, sos_obj_t obj)
{
	return __remove_key(sos, obj, NULL);
}

int sos_iter_obj_remove(sos_iter_t iter)
{
	return __remove_key(iter->sos, NULL, iter);
}

void sos_attr_set(sos_attr_t attr, sos_obj_t obj, void *value)
{
	attr->set_fn(attr, obj, value);
}

void *sos_value_get(sos_t sos, int attr_id, sos_obj_t obj)
{
	sos_attr_t attr = sos_value_by_id(sos, attr_id);
	if (!attr)
		/* SegV on a bad attribute id */
		kill(getpid(), SIGSEGV);
	return attr->get_fn(attr, obj);
}

void *sos_attr_get(sos_attr_t attr, sos_obj_t obj)
{
	return &obj->data[attr->data];
}


LIST_HEAD(__str_lst_head, __str_lst);
struct __str_lst {
	char *str;
	LIST_ENTRY(__str_lst) link;
};

void __str_lst_add(struct __str_lst_head *head, const char *str)
{
	struct __str_lst *sl = malloc(sizeof(*sl));
	sl->str = strdup(str);
	if (!sl->str)
		goto err0;
	LIST_INSERT_HEAD(head, sl, link);
	return;
err0:
	free(sl);
}

sos_t sos_destroy(sos_t sos)
{
	char *str = malloc(PATH_MAX+16);
	struct __str_lst_head head = {0};
	if (!str)
		return NULL; /* errno = ENOMEM should be set already */
	int i;
	/* object files */
	sprintf(str, "%s_sos.OBJ", sos->path);
	__str_lst_add(&head, str);
	sprintf(str, "%s_sos.PG", sos->path);
	__str_lst_add(&head, str);
	/* index files */
	for (i = 0; i < sos_meta(sos)->attr_cnt; i++) {
		sos_attr_t attr = sos->classp->attrs + i;
		if (attr->has_idx) {
			sprintf(str, "%s_%s.OBJ", sos->path, attr->name);
			__str_lst_add(&head, str);
			sprintf(str, "%s_%s.PG", sos->path, attr->name);
			__str_lst_add(&head, str);
		}
	}
	free(str);
	sos_close(sos, ODS_COMMIT_ASYNC);
	struct __str_lst *sl;
	int rc;
	while ((sl = LIST_FIRST(&head)) != NULL) {
		LIST_REMOVE(sl, link);
		rc = unlink(sl->str);
		if (rc)
			perror("unlink");
		free(sl->str);
		free(sl);
	}
	return sos;
}

void print_obj(sos_t sos, sos_obj_t obj, int attr_id)
{
	uint32_t t;
	uint32_t ut;
	uint32_t comp_id;
	uint64_t value;
	obj_key_t key = obj_key_new(1024);
	char tbuf[32];
	char kbuf[32];
	sos_blob_obj_t blob;
	const char *str;

	sos_value_key(sos, attr_id, obj, key);

	t = *(uint32_t *)sos_value_get(sos, 0, obj);
	ut = *(uint32_t *)sos_value_get(sos, 1, obj);
	sprintf(tbuf, "%d:%d", t, ut);
	sprintf(kbuf, "%02hhx:%02hhx:%02hhx:%02hhx",
		key->value[0],
		key->value[1],
		key->value[2],
		key->value[3]);
	comp_id = *(uint32_t *)sos_value_get(sos, 2, obj);
	value = *(uint64_t *)sos_value_get(sos, 3, obj);
	str = sos_value_get(sos, 4, obj);
	blob = sos_value_get(sos, 5, obj);
	printf("%11s %16s %8d %12lu %12s %*s\n",
	       kbuf, tbuf, comp_id, value, str, (int)blob->len, blob->data);
	obj_key_delete(key);
}

void sos_print_obj(sos_t sos, sos_obj_t obj, int attr_id)
{
	print_obj(sos, obj, attr_id);
}

/*** helper functions for swig ***/
int sos_get_attr_count(sos_t sos)
{
	return sos->classp->count;
}

enum sos_type_e sos_get_attr_type(sos_t sos, int attr_id)
{
	sos_attr_t attr = sos_value_by_id(sos, attr_id);
	if (!attr)
		return SOS_TYPE_UNKNOWN;
	return attr->type;
}

const char *sos_get_attr_name(sos_t sos, int attr_id)
{
	sos_attr_t attr = sos_value_by_id(sos, attr_id);
	if (!attr)
		return "N/A";
	return attr->name;
}

void sos_key_set_int32(sos_t sos, int attr_id, int32_t value, obj_key_t key)
{
	sos_value_key_set(sos, attr_id, &value, key);
}

void sos_key_set_int64(sos_t sos, int attr_id, int64_t value, obj_key_t key)
{
	sos_value_key_set(sos, attr_id, &value, key);
}

void sos_key_set_uint32(sos_t sos, int attr_id, uint32_t value, obj_key_t key)
{
	sos_value_key_set(sos, attr_id, &value, key);
}

void sos_key_set_uint64(sos_t sos, int attr_id, uint64_t value, obj_key_t key)
{
	sos_value_key_set(sos, attr_id, &value, key);
}

void sos_key_set_double(sos_t sos, int attr_id, double value, obj_key_t key)
{
	sos_value_key_set(sos, attr_id, &value, key);
}

int sos_verify_index(sos_t sos, int attr_id)
{
	sos_attr_t attr = sos_value_by_id(sos, attr_id);
	if (!sos_attr_has_index(attr))
		return 0;
	return obj_idx_verify(attr->oidx);
}

#endif
