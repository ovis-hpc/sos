/*
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2015 Sandia Corporation. All rights reserved.
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
#include <limits.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sos/sos.h>
#include "sos_priv.h"

int handle_partition_enable(sos_t sos, sos_config_t config);
int handle_partition_size(sos_t sos, sos_config_t config);
int handle_partition_period(sos_t sos,sos_config_t config);
int handle_partition_extend(sos_t sos, sos_config_t config);

static struct config_opt {
	const char *opt_name;
	int (*opt_handler)(sos_t sos, sos_config_t config);
} config_opts[] = {
	{ SOS_CONTAINER_PARTITION_ENABLE, handle_partition_enable },
	{ SOS_CONTAINER_PARTITION_SIZE, handle_partition_size },
	{ SOS_CONTAINER_PARTITION_PERIOD, handle_partition_period },
	{ SOS_CONTAINER_PARTITION_EXTEND, handle_partition_extend }
};

static int compare_opt(const void *a, const void *b)
{
	const struct config_opt *oa = a;
	const struct config_opt *ob = b;
	return strcmp(oa->opt_name, ob->opt_name);
}

static void option_handler(sos_t sos, sos_config_t config)
{
	struct config_opt *opt;
	struct config_opt config_opt;
	config_opt.opt_name = config->name;
	opt = bsearch(&config_opt, config_opts,
		      sizeof(config_opts) / sizeof(config_opts[0]),
		      sizeof(config_opts[0]),
		      compare_opt);
	if (opt)
		opt->opt_handler(sos, config);
}

int __sos_config_init(sos_t sos)
{
	sos_config_iter_t iter = sos_config_iter_new(sos->path);
	if (!iter)
		return ENOMEM;

	sos_config_t cfg;
	for (cfg = sos_config_first(iter); cfg; cfg = sos_config_next(iter))
		option_handler(sos, cfg);
	sos_config_iter_free(iter);
	return 0;
}

static void normalize_option_name(char *name)
{
	char *s;
	for (s = name; *s != '\0'; s++)
		if (islower(*s))
			*s = toupper(*s);
}

/**
 * \brief Set a container configuration option
 *
 * Container configuration options are used to manage the storage
 * management features of SOS, e.g. partitions, and partition
 * conditions. Container configuration options are persistent and only
 * need to be specified once.
 *
 * Sets the value of a SOS container option. Options include:
 * SOS_CONTAINER_PARTITION_ENABLE:
 *     Specifies if partition rotation is enabled. If enabled, and one
 *     of the conditions are met (size, period), a new partition is
 *     created and all new objects are placed in the new container. If
 *     partition rotate is disabled, and a size or duration condition
 *     is met, new object creation will fail.
 * SOS_CONTAINER_PARTITION_SIZE:
 *     Specifies the maximum size of a partition. When a partition
 *     reaches this size, new data overflow into the next partition.
 * SOS_CONTAINER_PARTITION_PERIOD:
 *     Specifies the duration in seconds of a partition. When the
 *     current time plus the 'start time' is reached, new data will
 *     overflow into the next partition. The 'star time' for a
 *     partition is Midnight.
 * SOS_CONTAINER_PARTITION_EXTEND_SIZE:
 *     When a partition must grow to accomodate new objects, this
 *     value specifies how much space to add.
 */
int sos_container_config_set(const char *path, const char *opt_name, const char *opt_value)
{
	char tmp_path[PATH_MAX];
	int rc;
	ODS_KEY(key);
	sos_obj_ref_t obj_ref;
	char *option_name;
	ods_key_t config_key;
	size_t key_len;
	sos_config_t config;
	ods_obj_t obj;
	ods_iter_t iter;

	if (!opt_name || !opt_value)
		return EINVAL;
	option_name = strdup(opt_name);
	if (!option_name)
		return ENOMEM;

	key_len = strlen(option_name) + 1;
	if (key_len > SOS_CONFIG_NAME_LEN) {
		free(option_name);
		return EINVAL;
	}
	normalize_option_name(option_name);

	/* Open the configuration ODS */
	sprintf(tmp_path, "%s/.__config", path);
	ods_t config_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!config_ods)
		goto err_0;

	/* Open the configuration object index */
	sprintf(tmp_path, "%s/.__config_idx", path);
	ods_idx_t config_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!config_idx)
		goto err_1;

	/* Allocate a new config object */
	obj = ods_obj_alloc(config_ods, sizeof(*config) + strlen(opt_value)+1);
	if (!obj)
		goto err_2;

	config_key = ods_key_alloc(config_idx, key_len);
	if (!config_key)
		goto err_3;
	ods_key_set(config_key, option_name, key_len);

	iter = ods_iter_new(config_idx);
	if (!iter)
		goto err_4;
	ods_key_set(&key, option_name, key_len);
	rc = ods_iter_find(iter, &key);
	if (!rc) {
		/* Delete the previous object */
		ods_key_t k = ods_iter_key(iter);
		sos_ref_reset(obj_ref);
		ods_idx_delete(config_idx, k, &obj_ref.idx_data);
		ods_ref_delete(config_ods, obj_ref.ref.obj);
	}
	config = SOS_CONFIG(obj);
	strcpy(config->name, option_name);
	strcpy(config->value, opt_value);
	obj_ref.ref.ods = 0;
	obj_ref.ref.obj = ods_obj_ref(obj);
	rc = ods_idx_insert(config_idx, config_key, obj_ref.idx_data);
	if (rc)
		goto err_5;
	ods_iter_delete(iter);
	ods_obj_put(config_key);
	ods_obj_put(obj);
	ods_close(config_ods, ODS_COMMIT_SYNC);
	ods_idx_close(config_idx, ODS_COMMIT_SYNC);
	free(option_name);
	return 0;
 err_5:
	ods_iter_delete(iter);
 err_4:
	ods_obj_delete(config_key);
	ods_obj_put(config_key);
 err_3:
	ods_obj_delete(obj);
	ods_obj_put(obj);
 err_2:
	ods_idx_close(config_idx, ODS_COMMIT_SYNC);
 err_1:
	ods_close(config_ods, ODS_COMMIT_SYNC);
 err_0:
	free(option_name);
	return errno;
}

char *sos_container_config_get(const char *path, const char *opt_name)
{
	char tmp_path[PATH_MAX];
	int rc;
	char *option_name;
	char *option_value;
	size_t key_len;
	sos_config_t config;
	ods_obj_t obj;
	SOS_KEY(config_key);
	sos_obj_ref_t config_ref;

	if (!opt_name)
		return NULL;
	option_name = strdup(opt_name);
	if (!option_name)
		return NULL;

	key_len = strlen(option_name) + 1;
	if (key_len > SOS_CONFIG_NAME_LEN)
		goto err_0;
	normalize_option_name(option_name);

	/* Open the configuration ODS */
	sprintf(tmp_path, "%s/.__config", path);
	ods_t config_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!config_ods)
		goto err_0;

	/* Open the configuration object index */
	sprintf(tmp_path, "%s/.__config_idx", path);
	ods_idx_t config_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!config_idx)
		goto err_1;

	ods_key_set(config_key, option_name, key_len);
	rc = ods_idx_find(config_idx, config_key, &config_ref.idx_data);
	if (rc)
		goto err_2;

	obj = ods_ref_as_obj(config_ods, config_ref.ref.obj);
	if (!obj)
		goto err_2;

	config = SOS_CONFIG(obj);
	option_value = strdup(config->value);
	if (!option_value)
		goto err_3;

	ods_obj_put(obj);
	ods_close(config_ods, ODS_COMMIT_ASYNC);
	ods_idx_close(config_idx, ODS_COMMIT_ASYNC);
	free(option_name);
	return option_value;

 err_3:
	ods_obj_put(obj);
 err_2:
	ods_idx_close(config_idx, ODS_COMMIT_SYNC);
 err_1:
	ods_close(config_ods, ODS_COMMIT_SYNC);
 err_0:
	free(option_name);
	return NULL;
}

int handle_partition_enable(sos_t sos, sos_config_t config)
{
	if (0 == strcasecmp(config->value, "yes"))
		sos->config.options |= SOS_OPTIONS_PARTITION_ENABLE;
	else if (0 == strcasecmp(config->value, "true"))
		sos->config.options |= SOS_OPTIONS_PARTITION_ENABLE;
	else
		sos->config.options &= ~SOS_OPTIONS_PARTITION_ENABLE;
	return 0;
}

long convert_time_units(const char *str)
{
	char *units;
	long value;

	value = strtol(str, &units, 0);
	if (value <= 0 || *units == '\0')
		return 0;
	switch (*units) {
	case 'm':
	case 'M':
		return value * 60;
	case 'h':
	case 'H':
		return value * 60 * 60;
	case 'd':
	case 'D':
		return value * 24 * 60 * 60;
	}
	return value;
}

long convert_size_units(const char *str)
{
	char *units;
	long value;

	value = strtol(str, &units, 0);
	if (value <= 0 || *units == '\0')
		return 0;
	switch (*units) {
	case 'k':
	case 'K':
		return value * 1024;
	case 'm':
	case 'M':
		return value * 1024 * 1024;
	case 'g':
	case 'G':
		return value * 1024 * 1024 * 1024;
	}
	return value;
}

int handle_partition_size(sos_t sos, sos_config_t config)
{
	sos->config.max_partition_size = convert_size_units(config->value);
	return 0;
}

int handle_partition_period(sos_t sos, sos_config_t config)
{
	sos->config.partition_period = convert_time_units(config->value);
	return 0;
}


int handle_partition_extend(sos_t sos, sos_config_t config)
{
	sos->config.partition_extend = convert_size_units(config->value);
	return 0;
}

sos_config_iter_t sos_config_iter_new(const char *path)
{
	char tmp_path[PATH_MAX];
	sos_config_iter_t iter = calloc(1, sizeof *iter);
	if (!iter)
		return NULL;

	/* Open the configuration ODS */
	sprintf(tmp_path, "%s/.__config", path);
	iter->config_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!iter->config_ods)
		goto err_0;

	/* Open the configuration object index */
	sprintf(tmp_path, "%s/.__config_idx", path);
	iter->config_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!iter->config_idx)
		goto err_1;

	iter->iter = ods_iter_new(iter->config_idx);
	if (!iter->iter)
		goto err_2;
	return iter;
 err_2:
	ods_idx_close(iter->config_idx, ODS_COMMIT_ASYNC);
 err_1:
	ods_close(iter->config_ods, ODS_COMMIT_ASYNC);
 err_0:
	free(iter);
	return NULL;
}

sos_config_t sos_config_first(sos_config_iter_t iter)
{
	int rc;
	if (iter->obj) {
		ods_obj_put(iter->obj);
		iter->obj = NULL;
	}
	rc = ods_iter_begin(iter->iter);
	if (rc)
		return NULL;
	sos_obj_ref_t obj_ref;
	obj_ref.idx_data = ods_iter_data(iter->iter);
	if (!obj_ref.ref.obj)
		return NULL;
	iter->obj = ods_ref_as_obj(iter->config_ods, obj_ref.ref.obj);
	return SOS_CONFIG(iter->obj);
}

sos_config_t sos_config_next(sos_config_iter_t iter)
{
	int rc;
	if (iter->obj) {
		ods_obj_put(iter->obj);
		iter->obj = NULL;
	}
	rc = ods_iter_next(iter->iter);
	if (rc)
		return NULL;
	sos_obj_ref_t obj_ref;
	obj_ref.idx_data = ods_iter_data(iter->iter);
	if (!obj_ref.ref.obj)
		return NULL;
	iter->obj = ods_ref_as_obj(iter->config_ods, obj_ref.ref.obj);
	return SOS_CONFIG(iter->obj);
}

void sos_config_iter_free(sos_config_iter_t iter)
{
	if (iter->obj)
		ods_obj_put(iter->obj);
	ods_iter_delete(iter->iter);
	ods_idx_close(iter->config_idx, ODS_COMMIT_ASYNC);
	ods_close(iter->config_ods, ODS_COMMIT_ASYNC);
	free(iter);
}

void sos_config_print(const char *path, FILE *fp)
{
	sos_config_t cfg;
	sos_config_iter_t iter = sos_config_iter_new(path);
	if (!iter)
		return;
	for (cfg = sos_config_first(iter); cfg; cfg = sos_config_next(iter))
		fprintf(fp, "%s=%s\n", cfg->name, cfg->value);
	sos_config_iter_free(iter);
}
