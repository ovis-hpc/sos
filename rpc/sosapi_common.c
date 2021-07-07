#include <sos/sos.h>
#include <errno.h>
#include <assert.h>
#include "sosapi.h"

/**
 * @brief Build an in memory SOS schema from a dsos_schema_spec
 *
 * @param spec The DSOS schema specification from the wire
 * @return sos_schema_t The SOS schema
 */
sos_schema_t dsos_schema_from_spec(struct dsos_schema_spec *spec)
{
	int attr_id, rc;
	sos_schema_t schema;

	schema = sos_schema_new(spec->name);
	if (!schema)
		return NULL;
	for (attr_id = 0; attr_id < spec->attrs.attrs_len; attr_id += 1)
	{
		dsos_schema_spec_attr *attr = &spec->attrs.attrs_val[attr_id];
		char *join_list[256];
		if (attr->type == SOS_TYPE_JOIN)
		{
			int i;
			assert(attr->size < 256);
			for (i = 0; i < attr->size; i++)
			{
				join_list[i] = attr->join_list.join_list_val[i].name;
			}
		}
		rc = sos_schema_attr_add(schema, attr->name, attr->type,
								 attr->size, join_list);
		if (rc)
		{
			errno = rc;
			goto err_1;
		}
		if (attr->indexed)
		{
			rc = sos_schema_index_add(schema, attr->name);
			if (rc)
			{
				errno = rc;
				goto err_1;
			}
			const char *idx_type = attr->idx_type;
			if (!idx_type || idx_type[0] == '\0')
				idx_type = "BXTREE";
			rc = sos_schema_index_modify(schema,
										 attr->name,
										 idx_type,
										 attr->key_type,
										 attr->idx_args);
			if (rc)
			{
				errno = rc;
				goto err_1;
			}
		}
	}
	return schema;
err_1:
	sos_schema_free(schema);
	return NULL;
}

static void attr_encode(sos_schema_t schema, dsos_schema_spec_attr *sattr, sos_attr_t attr)
{
	int i;
	sattr->name = strdup(sos_attr_name(attr));
	sattr->type = sos_attr_type(attr);
	sattr->size = sos_attr_size(attr);
	sattr->idx_type = strdup(sos_attr_idx_type(attr));
	sattr->key_type = strdup(sos_attr_key_type(attr));
	sattr->idx_args = strdup(sos_attr_idx_args(attr));

	sattr->indexed = sos_attr_is_indexed(attr);
	if (SOS_TYPE_JOIN != sos_attr_type(attr))
	{
		sattr->join_list.join_list_len = 0;
		sattr->join_list.join_list_val = NULL;
		return;
	}
	sos_array_t join_list = sos_attr_join_list(attr);
	assert(join_list);
	sattr->join_list.join_list_len = join_list->count;
	sattr->size = join_list->count;
	sattr->join_list.join_list_val = malloc(join_list->count * sizeof(dsos_attr_spec));
	assert(sattr->join_list.join_list_val);
	for (i = 0; i < join_list->count; i++)
	{
		uint32_t attr_id = join_list->data.uint32_[i];
		sos_attr_t join_attr = sos_schema_attr_by_id(schema, attr_id);
		sattr->join_list.join_list_val[i].name = strdup(sos_attr_name(join_attr));
		sattr->join_list.join_list_val[i].id = attr_id;
	}
}

void dsos_spec_free(struct dsos_schema_spec *spec)
{
	int attr_id;
	for (attr_id = 0; attr_id < spec->attrs.attrs_len; attr_id++) {
		dsos_schema_spec_attr *attr = &spec->attrs.attrs_val[attr_id];
		free(attr->name);
		free(attr->idx_type);
		free(attr->key_type);
		free(attr->idx_args);
		if (attr->type != SOS_TYPE_JOIN)
			continue;
		int i;
		for (i = 0; i < attr->join_list.join_list_len; i++) {
			free(attr->join_list.join_list_val[i].name);
		}
		free(attr->join_list.join_list_val);
	}
	free(spec->attrs.attrs_val);
	free(spec->name);
	free(spec);
}

/**
 * @brief Encode a sos_schema_t as a dsos_schema_spec
 *
 * On suceess a pointer to the dsos_schema_spec is returned. On failure
 * NULL is returned and \c errno is set to indicate the reason for failure.
 *
 * @param schema The schema handle
 * @return struct dsos_schema_spec*
 */
struct dsos_schema_spec *dsos_spec_from_schema(sos_schema_t schema)
{
	int attr_id, attr_count;
	struct dsos_schema_spec *spec;
	dsos_schema_spec_attr *attrs;
	spec = calloc(1, sizeof *spec);
	if (!spec)
		return NULL;
	attr_count = sos_schema_attr_count(schema);
	attrs = calloc(attr_count, sizeof(dsos_schema_spec_attr));
	if (!attrs)
		goto err_0;

	spec->attrs.attrs_val = attrs;
	spec->attrs.attrs_len = attr_count;
	spec->name = strdup(sos_schema_name(schema));
	if (!spec->name)
		goto err_1;
	sos_schema_uuid(schema, spec->uuid);
	for (attr_id = 0; attr_id < attr_count; attr_id++)
	{
		sos_attr_t attr = sos_schema_attr_by_id(schema, attr_id);
		attr_encode(schema, &attrs[attr_id], attr);
	}
	return spec;
err_1:
	free(attrs);
err_0:
	free(spec);
	return NULL;
}

struct dsos_schema_spec *dsos_schema_spec_dup(dsos_schema_spec *src_spec)
{
	int attr_id;
	dsos_schema_spec *dst_spec;
	dst_spec = calloc(1, sizeof *dst_spec);
	if (!dst_spec)
		return NULL;

	dst_spec->id = src_spec->id;
	dst_spec->name = strdup(src_spec->name);
	uuid_copy(dst_spec->uuid, src_spec->uuid);

	dst_spec->attrs.attrs_val = calloc(src_spec->attrs.attrs_len, sizeof(dsos_schema_spec_attr));
	dst_spec->attrs.attrs_len = src_spec->attrs.attrs_len;
	for (attr_id = 0; attr_id < src_spec->attrs.attrs_len; attr_id += 1)
	{
		dsos_schema_spec_attr *sattr = &src_spec->attrs.attrs_val[attr_id];
		dsos_schema_spec_attr *dattr = &dst_spec->attrs.attrs_val[attr_id];
		dattr->idx_args = strdup(sattr->idx_args);
		dattr->idx_type = strdup(sattr->idx_type);
		dattr->key_type = strdup(sattr->key_type);
		dattr->name = strdup(sattr->name);
		dattr->indexed = sattr->indexed;
		dattr->type = sattr->type;
		dattr->size = sattr->size;
		if (sattr->join_list.join_list_len) {
			int i;
			dattr->join_list.join_list_len = sattr->join_list.join_list_len;
			dattr->join_list.join_list_val = calloc(sattr->join_list.join_list_len, sizeof(dsos_attr_spec));
			for (i = 0; i < sattr->join_list.join_list_len; i++) {
				dattr->join_list.join_list_val[i].id = sattr->join_list.join_list_val[i].id;
				dattr->join_list.join_list_val[i].name = strdup(sattr->join_list.join_list_val[i].name);
			}
		} else {
			dattr->join_list.join_list_len = 0;
			dattr->join_list.join_list_val = NULL;
		}
	}
	return dst_spec;
}
