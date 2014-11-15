struct __idx_ods_arg {
	int rc;
	sos_attr_t attr;
};

void __idx_ods_rebuild_fn(ods_t ods, void *ptr, size_t sz, void *_arg)
{
	struct __idx_ods_arg *arg = _arg;
	if (arg->rc)
		return;
	sos_obj_t obj = ptr;
	if (obj->type != SOS_OBJ_TYPE_OBJ)
		return; /* skip non SOS object (e.g. attribute) */
	sos_t sos = arg->attr->sos;
	sos_attr_t attr = arg->attr;
	size_t attr_sz = sos_obj_attr_size(sos, attr->id, ptr);
	obj_ref_t obj_ref = ods_obj_ptr_to_ref(sos->ods, ptr);
	obj_key_t key = obj_key_new(attr_sz);
	if (!key) {
		arg->rc = ENOMEM;
		return;
	}
	sos_attr_key(attr, ptr, key);
	arg->rc = obj_idx_insert(attr->oidx, key, obj_ref);
	obj_key_delete(key);
}

int sos_rebuild_index(sos_t sos, int attr_id)
{
	int rc = 0;
	char *buff = malloc(PATH_MAX);
	if (!buff) {
		rc = ENOMEM;
		goto out;
	}
	sos_attr_t attr = sos_obj_attr_by_id(sos, attr_id);
	obj_idx_close(attr->oidx, ODS_COMMIT_ASYNC);
	attr->oidx = NULL;
	snprintf(buff, PATH_MAX, "%s_%s.OBJ", sos->path, attr->name);
	rc = unlink(buff);
	if (rc)
		goto out;
	snprintf(buff, PATH_MAX, "%s_%s.PG", sos->path, attr->name);
	rc = unlink(buff);
	if (rc)
		goto out;
	attr->oidx = init_idx(sos, O_CREAT|O_RDWR, 0660, attr);
	if (!attr->oidx) {
		fprintf(stderr, "sos: ERROR: Cannot initialize index: %s_%s\n",
			sos->path, attr->name);
		rc = -1;
		goto out;
	}

	struct __idx_ods_arg arg = {0, attr};

	ods_iter(sos->ods, __idx_ods_rebuild_fn, &arg);
	rc = arg.rc;

out:
	free(buff);
	return rc;
}

#define SOS_BAK_FMT ".%d"

static int __get_latest_backup(const char *sos_path)
{
	char *buff;
	int i, rc;
	struct stat st;

	buff = malloc(PATH_MAX);
	if (!buff)
		return -1;
	i = 1;
	while (1) {
		/* Check for sos_path.i.OBJ */
		snprintf(buff, PATH_MAX, "%s" SOS_BAK_FMT "_sos.OBJ", sos_path, i);
		rc = stat(buff, &st);
		if (rc)
			goto out;

		/* Check for sos_path.i.PG */
		snprintf(buff, PATH_MAX, "%s" SOS_BAK_FMT "_sos.PG", sos_path, i);
		rc = stat(buff, &st);
		if (rc)
			goto out;
		i++;
	}
out:
	free(buff);
	return i - 1;
}

static int __rename_obj_pg(const char *from, const char *to)
{
	size_t flen, tlen;
	char *_from, *_to;
	int rc = 0;

	flen = strlen(from);
	tlen = strlen(to);

	_from = malloc(flen + 5 + tlen + 5);
	if (!_from)
		return ENOMEM;
	_to = _from + flen + 5;
	strcpy(_from, from);
	strcpy(_to, to);

	strcpy(_from + flen, ".OBJ");
	strcpy(_to + tlen, ".OBJ");

	rc = rename(_from, _to);
	if (rc) {
		rc = errno;
		goto out;
	}

	strcpy(_from + flen, ".PG");
	strcpy(_to + tlen, ".PG");
	rc = rename(_from, _to);
	if (rc) {
		rc = errno;
		goto revert;
	}

	goto out;

revert:
	strcpy(_from + flen, ".OBJ");
	strcpy(_to + tlen, ".OBJ");
	/* trying to revert ... it is not guarantee though that this
	 * rename will be a success */
	rename(_to, _from);
out:
	free(_from);
	return rc;
}

static int __rename_sos(sos_class_t class, const char *from, const char *to)
{
	sos_attr_t attr;
	int attr_id;
	char *b0, *b1, *p0, *p1;
	int len0, len1, left0, left1;
	char *buff = malloc(PATH_MAX*2);
	int rc = 0;

	if (!buff) {
		rc = ENOMEM;
		goto out;
	}

	b0 = buff;
	b1 = buff + PATH_MAX;

	len0 = snprintf(b0, PATH_MAX, "%s", from);
	len1 = snprintf(b1, PATH_MAX, "%s", to);

	left0 = PATH_MAX - len0 - 1;
	left1 = PATH_MAX - len1 - 1;

	p0 = b0 + len0;
	p1 = b1 + len1;

	snprintf(p0, left0, "_sos");
	snprintf(p1, left1, "_sos");

	rc = __rename_obj_pg(b0, b1);
	if (rc)
		goto cleanup;

	for (attr_id = 0; attr_id < class->count; attr_id++) {
		attr = &class->attrs[attr_id];
		if (!attr->has_idx)
			continue;

		snprintf(p0, left0, "_%s", attr->name);
		snprintf(p1, left1, "_%s", attr->name);
		rc = __rename_obj_pg(b0, b1);
		if (rc == ENOENT) {
			rc = 0; /* ENOENT is OK */
			continue;
		}
		if (rc)
			goto rollback;
	}

	goto cleanup;

rollback:
	/* rollback only success attributes and main object store */
	attr_id--;
	while (attr_id > -1) {
		attr = &class->attrs[attr_id];
		if (!attr->has_idx)
			continue;

		snprintf(p0, left0, "_%s", attr->name);
		snprintf(p1, left1, "_%s", attr->name);
		__rename_obj_pg(b1, b0);
		attr_id--;
	}

	snprintf(p0, left0, "_sos");
	snprintf(p1, left1, "_sos");

	__rename_obj_pg(b1, b0);

cleanup:
	free(buff);
out:
	return rc;
}

/*
 * path is mutable, and expect to be able to append ".OBJ" or ".PG" at the end.
 */
static void __unlink_obj_pg(char *path)
{
	size_t len = strlen(path);
	strcpy(path + len, ".OBJ");
	unlink(path);
	strcpy(path + len, ".PG");
	unlink(path);
	path[len] = 0;
}

/*
 * path is mutable
 */
static void __unlink_sos(sos_class_t class, char *path)
{
	size_t len = strlen(path);
	int attr_id;
	sos_attr_t attr;
	strcpy(path + len, "_sos");
	__unlink_obj_pg(path);
	for (attr_id = 0; attr_id < class->count; attr_id++) {
		attr = &class->attrs[attr_id];
		if (!attr->has_idx)
			continue;
		sprintf(path + len, "_%s", attr->name);
		__unlink_obj_pg(path);
	}
	path[len] = 0;
}

int sos_chown(sos_t sos, uid_t owner, gid_t group)
{
	int rc;
	sos_attr_t attr;
	int attr_id;
	int attr_count = sos_get_attr_count(sos);
	for (attr_id = 0; attr_id < attr_count; attr_id++) {
		attr = sos_obj_attr_by_id(sos, attr_id);
		if (!attr->has_idx)
			continue;
		rc = obj_idx_chown(attr->oidx, owner, group);
		if (rc)
			return rc;
	}
	return ods_chown(sos->ods, owner, group);
}

/*** end helper functions ***/

static
sos_t __sos_rotate(sos_t sos, int N, int keep_index)
{
	sos_t new_sos = NULL;
	int M = __get_latest_backup(sos->path);
	char *buff;
	char *_a, *_b, *_tmp;
	size_t _len = strlen(sos->path);
	int i, attr_id, attr_count;
	struct stat st;
	int rc;

	rc = ods_stat(sos->ods, &st);
	if (rc)
		return NULL;

	buff = malloc(PATH_MAX * 2);
	if (!buff)
		return NULL;
	_a = buff;
	_b = _a + PATH_MAX;
	strcpy(_a, sos->path);
	strcpy(_b, sos->path);

	/* rename i --> i+1 */
	sprintf(_b + _len, SOS_BAK_FMT, M+1);
	for (i = M; i > -1; i--) {
		if (i)
			sprintf(_a + _len, SOS_BAK_FMT, i);
		else
			_a[_len] = '\0';

		rc = __rename_sos(sos->classp, _a, _b);
		if (rc)
			goto roll_back;

		_tmp = _a;
		_a = _b;
		_b = _tmp;
	}

	/* create new sos */
	new_sos = sos_open(sos->path, O_CREAT | O_RDWR, st.st_mode, sos->classp);
	if (!new_sos)
		goto roll_back;
	sos_chown(new_sos, st.st_uid, st.st_gid);

	/* close old sos and strip the indices, there's no going back from here */
	sos_close(sos, ODS_COMMIT_ASYNC);
	sos = new_sos;

	if (!keep_index) {
		attr_count = sos_get_attr_count(sos);
		for (attr_id = 0; attr_id < attr_count; attr_id++) {
			sos_attr_t attr = sos_obj_attr_by_id(sos, attr_id);
			if (!attr->has_idx)
				continue;
			sprintf(_a + _len, ".1_%s", attr->name);
			__unlink_obj_pg(_a);
		}
	}

	/* remove too-old backups */
	if (!N)
		goto out;

	for (i = N+1; i <= M + 1; i++) {
		sprintf(_a + _len, SOS_BAK_FMT, i);
		__unlink_sos(sos->classp, _a);
	}

	goto out;

roll_back:
	/* _sos rename rollback */
	i++;
	if (i)
		sprintf(_a + _len, SOS_BAK_FMT, i);
	else
		_a[_len] = '\0';
	while (i <= M) {
		/* rename i+1 -> i */
		sprintf(_b + _len, SOS_BAK_FMT, i+1);
		__rename_sos(sos->classp, _b, _a);

		i++;
		_tmp = _a;
		_a = _b;
		_b = _tmp;
	}
out:
	free(buff);
	return new_sos;
}

sos_t sos_rotate_i(sos_t sos, int N)
{
	return __sos_rotate(sos, N, 1);
}

sos_t sos_rotate(sos_t sos, int N)
{
	return __sos_rotate(sos, N, 0);
}

int sos_post_rotation(sos_t sos, const char *env_var)
{
	const char *cmd = getenv(env_var);
	char *buff;
	int rc = 0;
	if (!cmd)
		return ENOENT;
	buff = malloc(65536);
	if (!buff)
		return ENOMEM;
	snprintf(buff, 65536, "SOS_PATH=\"%s\" %s", sos->path, cmd);
	if (-1 == ovis_execute(buff))
		rc = ENOMEM;

	free(buff);
	return rc;
}

sos_t sos_reinit(sos_t sos, uint64_t sz)
{
	char *buff;
	int attr_id, attr_count;
	sos_attr_t attr;
	sos_t new_sos = NULL;
	sos_class_t class = NULL;
	mode_t mode = 0660;
	int rc = 0;
	struct stat _stat;

	if (!sz)
		sz = SOS_INITIAL_SIZE;

	buff = malloc(PATH_MAX);
	if (!buff)
		goto out;

	class = dup_class(sos->classp);
	if (!class)
		goto out;

	attr_count = sos->classp->count;
	for (attr_id = 0; attr_id < attr_count; attr_id++) {
		attr = sos_obj_attr_by_id(sos, attr_id);
		if (!attr->has_idx)
			continue;
		sprintf(buff, "%s_%s", sos->path, attr->name);
		__unlink_obj_pg(buff);
	}
	rc = ods_stat(sos->ods, &_stat);
	if (!rc)
		mode = _stat.st_mode;
	sprintf(buff, "%s_sos", sos->path);
	__unlink_obj_pg(buff);
	sos_close(sos, ODS_COMMIT_ASYNC);
	buff[strlen(buff) - 4] = 0;
	new_sos = sos_open_sz(buff, O_CREAT|O_RDWR, mode, class, sz);
out:
	free(buff);
	sos_class_free(class);
	return new_sos;
}

