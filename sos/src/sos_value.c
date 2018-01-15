/*
 * Copyright (c) 2014-2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2014-2015 Sandia Corporation. All rights reserved.
 *
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government.
 * Export of this program may require a license from the United States
 * Government.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the BSD-type
 * license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *      Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *      Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *      Neither the name of Sandia nor the names of any contributors may
 *      be used to endorse or promote products derived from this software
 *      without specific prior written permission.
 *
 *      Neither the name of Open Grid Computing nor the names of any
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *      Modified source versions must be plainly marked as such, and
 *      must not be misrepresented as being the original software.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Author: Tom Tucker tom at ogc dot us
 */
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sos/sos.h>
#include <errno.h>
#include <assert.h>
#include "sos_priv.h"

typedef int (*cmp_fn_t)(sos_value_t a, sos_value_t b);

static int INT16_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.int16_ - b->data->prim.int16_;
}

static int INT32_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.int32_ - b->data->prim.int32_;
}

static int INT64_cmp(sos_value_t a, sos_value_t b)
{
	if (a->data->prim.int64_ < b->data->prim.int64_)
		return -1;
	if (a->data->prim.int64_ > b->data->prim.int64_)
		return 1;
	return 0;
}

static int UINT16_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.uint16_ - b->data->prim.uint16_;
}

static int UINT32_cmp(sos_value_t a, sos_value_t b)
{
	return a->data->prim.uint32_ - b->data->prim.uint32_;
}

static int UINT64_cmp(sos_value_t a, sos_value_t b)
{
	if (a->data->prim.uint64_ < b->data->prim.uint64_)
		return -1;
	if (a->data->prim.uint64_ > b->data->prim.uint64_)
		return 1;
	return 0;
}

static int FLOAT_cmp(sos_value_t a, sos_value_t b)
{
	float res = a->data->prim.float_ - b->data->prim.float_;
	if (res > 0)
		return 1;
	else if (res < 0)
		return -1;
	return 0;
}

static int DOUBLE_cmp(sos_value_t a, sos_value_t b)
{
	double res = a->data->prim.double_ - b->data->prim.double_;
	if (res > 0)
		return 1;
	else if (res < 0)
		return -1;
	return 0;
}

static int LONG_DOUBLE_cmp(sos_value_t a, sos_value_t b)
{
	if (a->data->prim.long_double_ > b->data->prim.long_double_)
		return 1;
	if (a->data->prim.long_double_ < b->data->prim.long_double_)
		return -1;
	return 0;
}

static int TIMESTAMP_cmp(sos_value_t a, sos_value_t b)
{
	if (a->data->prim.timestamp_.time > b->data->prim.timestamp_.time)
		return 1;
	else if (a->data->prim.timestamp_.time < b->data->prim.timestamp_.time)
		return -1;
	return 0;
}

static int BYTE_ARRAY_cmp(sos_value_t a, sos_value_t b)
{
	int cmp_len = a->data->array.count;
	if (cmp_len > b->data->array.count)
		cmp_len = b->data->array.count;
	int res = memcmp(a->data->array.data.byte_, b->data->array.data.byte_, cmp_len);
	if (res == 0)
		return a->data->array.count - b->data->array.count;
	return res;
}

static int CHAR_ARRAY_cmp(sos_value_t a, sos_value_t b)
{
	return BYTE_ARRAY_cmp(a, b);
}

static int ARRAY_cmp(sos_value_t a, sos_value_t b)
{
	size_t a_len = sos_value_size(a);
	size_t b_len = sos_value_size(b);
	size_t cmp_len;
	if (a_len > b_len)
		cmp_len = b_len;
	else
		cmp_len = a_len;
	int res = memcmp(a->data->array.data.byte_, b->data->array.data.byte_, cmp_len);
	if (res == 0)
		return a_len - b_len;
	return res;
}

static cmp_fn_t cmp_fn_table[] = {
	[SOS_TYPE_INT16] = INT16_cmp,
	[SOS_TYPE_INT32] = INT32_cmp,
	[SOS_TYPE_INT64] = INT64_cmp,
	[SOS_TYPE_UINT16] = UINT16_cmp,
	[SOS_TYPE_UINT32] = UINT32_cmp,
	[SOS_TYPE_UINT64] = UINT64_cmp,
	[SOS_TYPE_FLOAT] = FLOAT_cmp,
	[SOS_TYPE_DOUBLE] = DOUBLE_cmp,
	[SOS_TYPE_LONG_DOUBLE] = LONG_DOUBLE_cmp,
	[SOS_TYPE_TIMESTAMP] = TIMESTAMP_cmp,
	[SOS_TYPE_CHAR_ARRAY] = CHAR_ARRAY_cmp,
	[SOS_TYPE_BYTE_ARRAY] = BYTE_ARRAY_cmp,
	[SOS_TYPE_INT16_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_INT32_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_INT64_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_UINT16_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_UINT32_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_UINT64_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_FLOAT_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_DOUBLE_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_LONG_DOUBLE_ARRAY] = ARRAY_cmp,
	[SOS_TYPE_OBJ_ARRAY] = ARRAY_cmp,
};

/**
 * \brief Compare two value
 *
 * Compares <tt>a</tt> and <tt>b</tt> and returns <0 if a < b, 0 if
 * a == b, and >0 if a > b
 *
 * \param a The lhs
 * \param b The rhs
 * \returns The result as described above
 */
int sos_value_cmp(sos_value_t a, sos_value_t b)
{
	if (a->attr->data->type < sizeof(cmp_fn_table)/sizeof(cmp_fn_table[0]))
		return cmp_fn_table[a->attr->data->type](a, b);
	return a == b;
}
static sos_value_t mem_value_init(sos_value_t val, sos_attr_t attr)
{
	memset(val, 0, sizeof(*val));
	val->attr = attr;
	val->data = &val->data_;
	return val;
}

sos_value_t sos_value_new()
{
	return calloc(1, sizeof(struct sos_value_s));
}

void sos_value_free(sos_value_t v)
{
	free(v);
}

/**
 * \brief Return a value for the specified object's attribute.
 *
 * This function returns a sos_value_t for an object's attribute.
 * The reference on this value should be dropped with sos_value_put()
 * when the application is finished with the value.
 *
 * \param obj The object handle
 * \param attr The attribute handle
 * \returns The value of the object's attribute.
 */
sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr)
{
	sos_value_t value = sos_value_new();
	if (value)
		value = sos_value_init(value, obj, attr);
	return value;
}

static sos_value_t __sos_join_value_init(sos_value_t val, sos_obj_t obj, sos_attr_t attr)
{
	struct sos_value_s *v_;
	sos_value_t *v;
	struct sos_value_s _v_[MAX_JOIN_ATTRS];
	sos_value_t _v[MAX_JOIN_ATTRS];
	ods_key_comp_t comp;
	size_t comp_len;
	struct sos_array_s *data;
	int i, count, join_id;
	sos_schema_t schema = sos_attr_schema(attr);
	size_t size;
	sos_attr_t join_attr;

	assert(schema);
	assert(sos_attr_type(attr) == SOS_TYPE_JOIN);

	count = attr->ext_ptr->count;
	if (count <= MAX_JOIN_ATTRS) {
		v_ = _v_;
		v = _v;
	} else {
		v_ = calloc(count, sizeof *v_);
		if (!v_)
			return NULL;
		v = calloc(count, sizeof *v);
		if (!v)
			return NULL;
	}

	size = 0;
	for (i = 0; i < count; i++) {
		join_id = attr->ext_ptr->data.uint32_[i];
		join_attr = sos_schema_attr_by_id(schema, join_id);
		if (!join_attr) {
			errno = EINVAL;
			goto err;
		}
		v[i] = sos_value_init(&v_[i], obj, join_attr);
		if (!v[i]) {
			errno = ENOMEM;
			goto err;;
		}
		size += sos_value_size(v[i]) + sizeof(comp->value.str);
	}

	if (size > (sizeof(val->data_) + sizeof(*data))) {
		data = malloc(sizeof(*data) + size);
		if (!data) {
			errno = ENOMEM;
			return NULL;
		}
	} else {
		data = (sos_array_t)&val->data_;
	}

	data->count = size;
	val->attr = attr;
	val->obj = NULL;
	val->data = (sos_value_data_t)data;

	comp = (ods_key_comp_t)data->data.byte_;

	for (i = 0; i < count; i++) {
		comp = __sos_set_key_comp(comp, v[i], &comp_len);
		sos_value_put(v[i]);
	}
	if (count > MAX_JOIN_ATTRS) {
		free(v_);
		free(v);
	}
	return val;
 err:
	while (i) {
		sos_value_put(v[i]);
		i--;
	}
	if (count > MAX_JOIN_ATTRS) {
		free(v_);
		free(v);
	}
	return NULL;
}

/**
 * \brief Initialize a value with an object's attribute data
 *
 * The returned value has a reference on the parameter object <tt>obj<tt>. The
 * caller must call sos_value_put() when the value is no longer in use.
 *
 * \param val Pointer to the value to be initialized
 * \param obj The object handle
 * \param attr The attribute handle
 * \retval The value handle
 */
sos_value_t sos_value_init(sos_value_t val, sos_obj_t obj, sos_attr_t attr)
{
	sos_obj_t ref_obj;
	sos_value_data_t ref_val;
	if (!obj)
		return mem_value_init(val, attr);

	if (sos_attr_type(attr) == SOS_TYPE_JOIN)
		return __sos_join_value_init(val, obj, attr);

	val->attr = attr;
	ref_val = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];
	if (!sos_attr_is_array(attr)) {
		val->obj = sos_obj_get(obj);
		val->data = ref_val;
		goto out;
	}
	ref_obj = sos_ref_as_obj(obj->sos, ref_val->prim.ref_);
	if (ref_obj) {
		val->obj = ref_obj; /* ref from sos_ref_as_obj */
		val->data = (sos_value_data_t)&SOS_OBJ(ref_obj->obj)->data[0];
	} else {
		val = NULL;
	}
 out:
	return val;
}

sos_value_t sos_value_copy(sos_value_t dst, sos_value_t src)
{
	dst->attr = src->attr;
	if (src->obj) {
		dst->obj = sos_obj_get(dst->obj);
		dst->data = src->data;
		return dst;
	}
	if (!sos_value_is_array(src)) {
		memcpy(&dst->data_, &src->data_, sizeof(src->data_));
		dst->data = &dst->data_;
		return dst;
	}
	size_t sz = sos_value_size(src);
	if (sz > sizeof(union sos_value_data_u) - sizeof(struct sos_array_s)) {
		dst->data = malloc(sz + sizeof(struct sos_array_s));
		if (!dst->data)
			return NULL;
	} else {
		dst->data = &dst->data_;
	}
	dst->data->array.count = src->data->array.count;
	memcpy(dst->data->array.data.byte_, src->data->array.data.byte_, sz);
	return dst;
}

/**
 * \brief Returns a pointer into the memory of an object
 *
 * This function returns a pointer directly into the memory of an
 * object. The \t attr parameter specifies which attribute in the
 * object is to be accessed.
 *
 * If the attribute is an array, the array object to which the
 * attribute refers is returned in the arr_obj parameter if
 * provided. This is because the memory pointed to by the return value
 * has a reference on the array object and the caller is responsible
 * for releasing that reference when the memory is no longer in use.
 *
 * If the \t arr_obj pointer is NULL and the \t attr parameter is an
 * array, this function will assert.
 *
 * \param obj The object handle
 * \param attr The attribute handle
 * \param arr_obj Pointer to a sos_obj_t handle.
 */
sos_value_data_t sos_obj_attr_data(sos_obj_t obj, sos_attr_t attr, sos_obj_t *arr_obj)
{
	sos_obj_t ref_obj;
	sos_value_data_t ref_val = NULL;

	if (arr_obj)
		*arr_obj = NULL;

	ref_val = (sos_value_data_t)&obj->obj->as.bytes[attr->data->offset];
	if (!sos_attr_is_array(attr))
		return ref_val;

	ref_obj = sos_ref_as_obj(obj->sos, ref_val->prim.ref_);
	if (ref_obj) {
		/* If this is an array object and the caller did not
		 * provide the arr_obj parameter, the object will be
		 * leaked */
		assert(arr_obj);
		*arr_obj = ref_obj; /* ref from sos_ref_as_obj */
		ref_val = (sos_value_data_t)&SOS_OBJ(ref_obj->obj)->data[0];
	}

	return ref_val;
}

/**
 * \brief Set an object value from a buffer
 *
 * Set the value from an untyped void buffer. If the buflen is too
 * large to fit, only sos_value_size() bytes will be written.
 *
 * \param val    The value handle
 * \param buf    The buffer containing the data
 * \param buflen The number of bytes to write from the buffer
 * \retval The number of bytes written
 */
size_t sos_value_memcpy(sos_value_t val, void *buf, size_t buflen)
{
	void *dst;
	if (buflen > sos_value_size(val))
		buflen = sos_value_size(val);
	if (!sos_attr_is_array(val->attr))
		dst = val->data;
	else
		dst = &val->data->array.data.byte_[0];
	memcpy(dst, buf, buflen);
	return buflen;
}

/**
 * \brief Drop a reference on a value object
 *
 * \param value The value handle.
 */
void sos_value_put(sos_value_t value)
{
	if (!value)
		return;
	if (value->obj) {
		sos_obj_put(value->obj);
		value->obj = NULL;
	} else {
		assert(value->data);
		if (value->data != &value->data_) {
			free(value->data);
			value->data = NULL;
		}
	}
}

/**
 * \brief Initialize a value with an object's attribute data
 *
 * Returns an object value handle for the specified attribute
 * name. If the <tt>attr_id</tt> parameter is non-null, the parameter
 * is filled in with associated attribute id.
 *
 * \param value Pointer to the value to be initialized
 * \param schema The schema handle
 * \param obj The object handle
 * \param name The attribute name
 * \param attr_id A pointer to the attribute id
 * \retval Pointer to the sos_value_t handle
 * \retval NULL if the specified attribute does not exist.
 */
sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
			      const char *name, int *attr_id)
{
	sos_attr_t attr = sos_schema_attr_by_name(schema, name);
	if (!attr)
		return NULL;
	return sos_value_init(value, obj, attr);
}

/**
 * \brief Initialize a value with an object's attribute data
 *
 * Returns the sos_value_t for the attribute with the specified
 * id.
 *
 * \param value Pointer to the value to be initialized
 * \param obj		The SOS object handle
 * \param attr_id	The Id for the attribute.
 * \retval Pointer to the sos_value_t handle
 * \retval NULL if the specified attribute does not exist.
 */
sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id)
{
	sos_attr_t attr = sos_schema_attr_by_id(obj->schema, attr_id);
	if (!attr)
		return NULL;
	return sos_value_init(value, obj, attr);
}

/**
 * \brief Format a value as a string
 *
 * \param v   The value handle
 * \param str Pointer to the string to receive the formatted value
 * \param len The size of the string in bytes.
 * \returns A pointer to the str argument or NULL if there was a
 *          formatting error.
 */
const char *sos_value_to_str(sos_value_t v, char *str, size_t len)
{
	return v->attr->to_str_fn(v, str, len);
}

/**
 * \brief Set the value from a string
 *
 * \param v The value handle
 * \param str The input string value to parse
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 The string was successfully parsed and the value set
 * \retval EINVAL The string was incorrectly formatted for this value
 *                type.
 */
int sos_value_from_str(sos_value_t v, const char *str, char **endptr)
{
	return v->attr->from_str_fn(v, str, endptr);
}

/**
 * \brief Return the length of the string required for the value
 *
 * This function returns the size of the string required to hold the
 * attribute value if formatted as a string. This function is useful
 * when allocating buffers used with the sos_obj_attr_to_str()
 * function. The returned value does not include the byte required
 * to contain the terminating '\0'.
 *
 * \param v The value handle
 * \returns The size of the string in bytes.
 */
size_t sos_value_strlen(sos_value_t v)
{
	return v->attr->strlen_fn(v);
}

