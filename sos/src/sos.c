/* -*- c-basic-offset : 8 -*-
 * Copyright (c) 2012-2021 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2020 Sandia Corporation. All rights reserved.
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
/**
 * \mainpage Introduction
 *
 * The Scalable Object Storage (SOS) is a high performance storage
 * engine designed to efficiently manage structured data on persistent
 * media.
 *
 * \section cont Container
 *
 * An instance of a SOS Object Store is called a Container. A
 * Container is identified by a name that looks like a POSIX Filesytem
 * path. This allows Containers to be organized into a hierarchical
 * name space. This is a convenience and does not mean that a
 * Container is necessarily stored in a Filesytem.
 *
 * See \ref container_overview for more information.

 * \section part Partition
 *
 * In order to facilitate management of the storage consumed by a
 * Container, a Container is divided up into one or more
 * Partitions. Paritions contain the objects that are created in the
 * Container. The purpose of a Partition is to allow subsets of a
 * Container's objects to be migrated from primary storage to secondary
 * storage.
 *
 * See the \ref partitions section for more information
 *
 * \section schema Schema
 *
 * Schemas define the format of Objects and are logically an Object's
 * "type." There can be any number of Schema in the Container such
 * that a single Container may contain Objects of many different
 * types. The Container has a directory of Schemas. When Objects are
 * created, the Schema handle is specified to inform the object store
 * of the size and format of the object and whether or not one or more
 * of its attributes has an Index.
 *
 * See \ref schema_overview for more information.
 *
 * \section object Object
 *
 * An Object is a collection of Attributes. An Attribute has a Name
 * and a Type. There are built-in types for an Attribute and
 * user-defined types. The built in types include the familiar
 * <tt>int</tt>, <tt>long</tt>, <tt>double</tt> types as well as
 * arrays of these types. A special Attribute type is
 * <tt>SOS_TYPE_OBJ</tt>, which is a <tt>Reference</tt> to another
 * Object. This allows complex data structures like linked lists to be
 * implemented in the Container.
 *
 * The user-defined types are Objects.
 *
 * \section index_intro Indices
 *
 * An Index is a named, ordered collection of Key/Value references to
 * Objects. Their purpose is to quickly find an Object in a container
 * based on a Key. Indexes can be associated with a Schema or be
 * independent of a Schema, for example, allowing a single Index to
 * refer to objects of different types. If an Index is associated with
 * a Schema Attribute, all management and insertion is handled
 * automatically by the sos_obj_index() function.
 *
 * Indexes that are not part of a Schema, i.e. not associated with an
 * indexed attribute, are managed by the application; including the
 * creation of keys, and insertion of Objects into the index.
 *
 * See the \ref indices section for more information.
 */

#define _GNU_SOURCE
#include <sys/queue.h>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <libgen.h>
#include <errno.h>
#include <assert.h>
#include <ftw.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

#ifndef DOXYGEN
LIST_HEAD(cont_list_head, sos_container_s)
cont_list;
#endif
pthread_mutex_t cont_list_lock;
pthread_mutex_t _sos_log_lock;
static sos_obj_ref_t NULL_REF = {
	.ref = { { 0 }, 0 }
};

char *sos_obj_ref_to_str(sos_obj_ref_t ref, sos_obj_ref_str_t str)
{
	char uuid_str[37];
	uuid_unparse_lower(ref.ref.part_uuid, uuid_str);
	(void)snprintf(str, sizeof(sos_obj_ref_str_t), "%s@%p", uuid_str, (void *)ref.ref.obj);
	return str;
}

int sos_obj_ref_from_str(sos_obj_ref_t ref, const char *value, char **endptr)
{
	ods_ref_t obj_ref;
	sos_obj_ref_str_t part_ref;
	int cnt;
	int match = sscanf(value, "%s@%lx%n", part_ref, &obj_ref, &cnt);
	if (match < 2)
		return EINVAL;
	if (endptr)
		*endptr = (char *)(value + cnt);
	uuid_parse(part_ref, ref.ref.part_uuid);
	ref.ref.obj = obj_ref;
	return 0;
}

/**
 * \page container_overview Containers
 *
 * A SOS Container groups Partitions, Schema, Objects, and Indices
 * together into a single namespace. The root of the namespace is the
 * Container's name. Containers are created with the
 * sos_container_open() function. SOS implements the POSIX security
 * model. When a Container is created, it inherits the owner and group
 * of the process that created the container. The sos_container_open()
 * function takes an <tt>o_mode</tt> parameter that identifies the
 * standard POSIX umask to specify RO/RW access for owner/group/other.
 * The user/group of the process opening the container must
 * have adequate permission to provide the requested R/W access. The
 * sos_container_open() function returns a <tt>sos_t</tt> container handle that
 * is used in subsequent SOS API.
 *
 * Changes to a container are opportunistically commited to stable
 * storage. An application can initiate a commit to storage with the
 * sos_container_commit() function.
 *
 * The SOS Container API include the following:
 *
 * - sos_container_open() Open/create a Container
 * - sos_container_close() Close a container
 * - sos_container_commit() Commit a Container's contents to stable storage
 * - sos_container_info() - Print Container information to a FILE pointer
 * - sos_container_lock_info() - Print Container lock information to a FILE pointer
 * - sos_container_lock_cleanup() - Release locks held by no process
 */

/* This function effectively implements 'mkdir -p' */
int __sos_make_all_dir(const char *inp_path, mode_t omode)
{
	struct stat sb;
	mode_t oumask;
	int last, retval;
	char *p, *path;

	p = path = strdup(inp_path);
	if (!p) {
		errno = ENOMEM;
		return 1;
	}

	oumask = 0;
	retval = 0;
	if (p[0] == '/')
		++p;

	oumask = umask(0);
	for (last = 0; !last ; ++p) {
		if (p[0] == '\0')
			last = 1;
		else if (p[0] != '/')
			continue;
		*p = '\0';
		if (!last && p[1] == '\0')
			last = 1;
		if (mkdir(path, omode) < 0) {
			if (errno == EEXIST || errno == EISDIR) {
				if (stat(path, &sb) < 0) {
					retval = 1;
					break;
				} else if (!S_ISDIR(sb.st_mode)) {
					if (last)
						errno = EEXIST;
					else
						errno = ENOTDIR;
					retval = 1;
					break;
				}
			} else {
				retval = 1;
				break;
			}
		}
		if (!last)
			*p = '/';
	}
	(void)umask(oumask);
	free(path);
	return retval;
}

struct idx_ent_s {
	char name[SOS_INDEX_NAME_LEN];
	sos_index_t idx;
	sos_iter_t iter;
	struct ods_rbn rbn;
};

/** \defgroup container SOS Storage Containers
 * @{
 */

/** \defgroup container SOS Storage Containers
 * @{
 */

/**
 * \brief Create a Container
 *
 * This interface is deprecated. New software should use sos_container_open()
 *
 * Creates a SOS container. The o_flags and o_mode parameters accept
 * the same values and have the same meaning as the corresponding
 * parameters to the open() system call.
 *
 * Like the POSIX \c creat() system call, this function is racy. New software
 * should use use the \c sos_container_open() function.
 *
 * \param path		Pathname for the Container.
 * \param o_mode	The file mode for the Container.
 * \retval 0		The container was successfully created.
 * \retval EINVAL	A parameter was invalid
 * \retval EPERM	The user has insufficient permission
 * \retval EEXIST	A container already exists at the specified path
 */
int sos_container_new(const char *path, int o_mode)
{
	sos_t sos = sos_container_open(path, SOS_PERM_CREAT, o_mode);
	if (!sos)
		return errno;
	sos_container_close(sos, SOS_COMMIT_SYNC);
	return 0;
}

/* Must be called holding the SOS container file lock */
static int __sos_container_new(const char *path, sos_perm_t *p_perm, int o_mode)
{
	char tmp_path[PATH_MAX];
	char real_path[PATH_MAX];
	int rc;
	int x_mode;
	ods_t ods;
	sos_perm_t o_perm = *p_perm;

	if (0 == (o_perm & (SOS_BE_LSOS | SOS_BE_MMOS))) {
		char *bes = getenv("SOS_DEFAULT_BACKEND");
		if (bes) {
			if (0 == strcasecmp(bes, "lsos")) {
				o_perm |= SOS_BE_LSOS;
			} else {
				o_perm |= SOS_BE_MMOS;
			}
		} else {
			o_perm |= SOS_BE_MMOS;
		}
	}

	x_mode = o_mode;
	if (x_mode & (S_IWGRP | S_IRGRP))
		x_mode |= S_IXGRP;
	if (x_mode & (S_IWUSR | S_IRUSR))
		x_mode |= S_IXUSR;
	if (x_mode & (S_IWOTH | S_IROTH))
		x_mode |= S_IXOTH;

	rc = __sos_make_all_dir(path, x_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}
	path = realpath(path, real_path);
	if (!path) {
		rc = errno;
		goto err_1;
	}

	/* Create the ODS to contain configuration objects */
	sprintf(tmp_path, "%s/.__config", path);
	ods = ods_open(tmp_path, o_perm | ODS_PERM_CREAT | ODS_PERM_RW, o_mode);
	ods_close(ods, ODS_COMMIT_SYNC);
	if (!ods)
		goto err_1;

	/* Create the configuration object index */
	sprintf(tmp_path, "%s/.__config_idx", path);
	rc = ods_idx_create(tmp_path, o_perm, o_mode, "BXTREE", "STRING", NULL);
	if (rc)
		goto err_2;

	/* Create the ODS to contain partition objects */
	sprintf(tmp_path, "%s/.__part", path);
	ods_t part_ods = ods_open(tmp_path, o_perm | ODS_PERM_CREAT | ODS_PERM_RW, o_mode);
	if (!part_ods)
		goto err_3;

	ods_obj_t udata = ods_get_user_data(part_ods);
	if (!udata) {
		rc = errno;
		ods_close(part_ods, ODS_COMMIT_ASYNC);
		goto err_4;
	}
	SOS_PART_REF_UDATA(udata)->signature = SOS_PART_REF_SIGNATURE;
	SOS_PART_REF_UDATA(udata)->primary = 0;
	SOS_PART_REF_UDATA(udata)->head = 0;
	SOS_PART_REF_UDATA(udata)->tail = 0;
	SOS_PART_REF_UDATA(udata)->lock = 0;
	ods_obj_update(udata);
	ods_obj_put(udata);
	ods_close(part_ods, ODS_COMMIT_SYNC);

	/* Create the ODS to contain the schema objects */
	sprintf(tmp_path, "%s/.__schemas", path);
	ods_t schema_ods = ods_open(tmp_path, o_perm | ODS_PERM_CREAT | O_RDWR, o_mode);
	if (!schema_ods)
		goto err_4;
	/* Initialize the schema dictionary */
	udata = ods_get_user_data(schema_ods);
	if (!udata) {
		rc = errno;
		ods_close(schema_ods, ODS_COMMIT_ASYNC);
		goto err_5;
	}
	SOS_SCHEMA_UDATA(udata)->signature = SOS_SCHEMA_SIGNATURE;
	SOS_SCHEMA_UDATA(udata)->version = SOS_LATEST_VERSION;
	ods_obj_update(udata);
	ods_obj_put(udata);
	ods_close(schema_ods, ODS_COMMIT_ASYNC);

	/* Create the index to look up the schema names */
	sprintf(tmp_path, "%s/.__schema_idx", path);
	rc = ods_idx_create(tmp_path, o_perm, o_mode, "BXTREE", "STRING", NULL);
	if (rc)
		goto err_5;

	/* Create the ODS to contain the index objects */
	sprintf(tmp_path, "%s/.__index", path);
	ods_t idx_ods = ods_open(tmp_path, o_perm | SOS_PERM_RW | SOS_PERM_CREAT, o_mode);
	if (!idx_ods)
		goto err_6;

	/* Initialize the index dictionary */
	udata = ods_get_user_data(idx_ods);
	if (!udata) {
		rc = errno;
		ods_close(idx_ods, ODS_COMMIT_ASYNC);
		goto err_7;
	}
	SOS_IDXDIR_UDATA(udata)->signature = SOS_IDXDIR_SIGNATURE;
	SOS_IDXDIR_UDATA(udata)->lock = 0;
	ods_obj_update(udata);
	ods_obj_put(udata);
	ods_close(idx_ods, ODS_COMMIT_ASYNC);

	/* Create the index to look up the indexes */
	sprintf(tmp_path, "%s/.__index_idx", path);
	rc = ods_idx_create(tmp_path, o_perm, o_mode, "BXTREE", "STRING", NULL);
	if (rc)
		goto err_7;
	*p_perm = o_perm;
	const char *be_type = "MMOS";
	if (o_perm & SOS_BE_LSOS)
		be_type = "LSOS";
	sos_container_config_set(path, "BACKEND_TYPE", be_type);
	return 0;
err_7:
	sprintf(tmp_path, "%s/.__index", path);
	ods_destroy(tmp_path);
err_6:
	sprintf(tmp_path, "%s/.__schemas_idx", path);
	ods_destroy(tmp_path);
err_5:
	sprintf(tmp_path, "%s/.__schemas", path);
	ods_destroy(tmp_path);
err_4:
	sprintf(tmp_path, "%s/.__part", path);
	ods_destroy(tmp_path);
err_3:
	sprintf(tmp_path, "%s/.__config_idx", path);
	ods_destroy(tmp_path);
err_2:
	sprintf(tmp_path, "%s/.__config", path);
	ods_destroy(tmp_path);
err_1:
	rmdir(path);
	errno = rc; /* rmdir will stomp errno */
err_0:
	return rc;
}

const char *sos_container_path(sos_t sos)
{
	return sos->path;
}

sos_perm_t sos_container_perm(sos_t sos)
{
	return sos->o_perm;
}

int sos_container_mode(sos_t sos)
{
	return sos->o_mode;
}

/**
 * \brief Delete storage associated with a Container
 *
 * Removes all resources associated with the Container. The sos_t
 * handle must be provided (requiring an open) because it is necessary
 * to know the associated Indices in order to be able to know the
 * names of the associated files. sos_destroy() will also close \c sos, as the
 * files should be closed before begin removed.
 *
 * \param c	The container handle
 * \retval 0	The container was deleted
 * \retval EPERM The user has insufficient privilege
 * \retval EINUSE The container is in-use by other clients
 */
int sos_container_delete(sos_t c)
{
	return ENOSYS;
}

/**
 * \brief Flush outstanding changes to persistent storage
 *
 * This function commits the index changes to stable storage. If
 * SOS_COMMIT_SYNC is specified in the flags parameter, the function
 * will wait until the changes are commited to stable stroage before
 * returning.
 *
 * \param sos	The SOS container handle
 * \param flags	The commit flags
 */
int sos_container_commit(sos_t sos, sos_commit_t flags)
{
	sos_schema_t schema;
	sos_attr_t attr;
	int commit;

	if (flags == SOS_COMMIT_SYNC)
		commit = ODS_COMMIT_SYNC;
	else
		commit = ODS_COMMIT_ASYNC;

	/* Commit the schema idx and ods */
	ods_commit(sos->schema_ods, commit);
	ods_idx_commit(sos->schema_idx, commit);

	/* Commit the object ods */
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry)
	if (part->obj_ods)
		ods_commit(part->obj_ods, commit);

	/* Commit all the attribute indices */
	LIST_FOREACH(schema, &sos->schema_list, entry) {
		if (schema->state != SOS_SCHEMA_OPEN)
			continue;
		TAILQ_FOREACH(attr, &schema->attr_list, entry) {
			if (sos_attr_index(attr))
				sos_index_commit(sos_attr_index(attr), commit);
		}
	}
	return 0;
}

/**
 * @brief Begin a transaction on a SOS container
 *
 * Attempts to begin a transaction boundary on a container. The \c
 * timeout parameter specifies how long to wait before giving up. If a
 * transaction could not be acquired by the specified timeout,
 * ETIMEDOUT is returned.  Otherwise zero is returned. A NULL
 * \c timeout indicates that the function will wait forever.
 *
 * @param sos The container handle
 * @param timeout Pointer to a timespec structure indicating
 * @returns 0 if the transaction was acquired or ETIMEDOUT if a
 *            transaction could not be acquired within the specified
 *            timeout.
 */
int sos_begin_x_wait(sos_t sos, struct timespec *ts)
{
	if (!sos->part_ref_ods)
		return EINVAL;
	return ods_begin_x(sos->part_ref_ods, ts);
}

/**
 * @brief Begin a transaction on a SOS container
 *
 * Begin a transaction boundary on a container. This function
 * will wait indefinitely for the container to become available and
 * is equivalent to sos_begin_x_wait(sos, NULL)
 *
 * @param sos The container handle
 * @returns 0
 */
int sos_begin_x(sos_t sos)
{
	if (!sos->part_ref_ods)
		return EINVAL;
	return ods_begin_x(sos->part_ref_ods, NULL);
}

void sos_end_x(sos_t sos)
{
	ods_end_x(sos->part_ref_ods);
}

/**
 * @brief Return storage utilization information about an index
 *
 * @param index The index handle
 * @param fp Pointer to a FILE into which data will be dumped
 */
void sos_index_info(sos_index_t index, FILE *fp)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	ods_idx_info(iref->idx, fp);
	ods_info(ods_idx_ods(iref->idx), fp, ODS_INFO_ALL);
}

/**
 * @brief Verify the internal consistency of an index
 *
 * @param index The index handle
 * @param fp A FILE pointer into which error information is reported
 * @returns 0 The index is consistent
 * @returns Consult \c fp for error message(s)
 */
int sos_index_verify(sos_index_t index, FILE *fp)
{
	ods_idx_ref_t iref = LIST_FIRST(&index->active_idx_list);
	return ods_idx_verify(iref->idx, fp);
}

int print_schema(struct ods_rbn *n, void *fp_, int level)
{
	FILE *fp = fp_;
	sos_attr_t attr;

	sos_schema_t schema = container_of(n, struct sos_schema_s, name_rbn);
	sos_schema_print(schema, fp);

	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (sos_attr_index(attr))
			sos_index_info(sos_attr_index(attr), fp);
	}
	return 0;
}

/**
 * \brief Print container information
 *
 * Prints information about the container to the specified FILE pointer.
 *
 * \param sos	The container handle
 * \param fp	The FILE pointer
 */
void sos_container_info(sos_t sos, FILE *fp)
{
	if (!fp)
		fp = stdout;
	ods_rbt_traverse(&sos->schema_name_rbt, print_schema, fp);
	ods_idx_info(sos->schema_idx, fp);
	ods_info(ods_idx_ods(sos->schema_idx), fp, ODS_INFO_ALL);
	ods_info(sos->schema_ods, fp, ODS_INFO_ALL);
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (part->obj_ods)
			ods_info(part->obj_ods, fp, ODS_INFO_ALL);
	}
}

static int show_locks(const char *path, const struct stat *sb,
			int typeflags, struct FTW *ftw)
{
	size_t len;
	char tmp_path[PATH_MAX];

	strncpy(tmp_path, path, PATH_MAX);
	len = strlen(tmp_path);
	if (strcmp(&tmp_path[len-3], ".BE"))
		return 0;
	/* strip the .PG, ods_lock_info will append it */
	tmp_path[len-3] = '\0';
	ods_lock_info(tmp_path, stdout);
	return 0;
}

static int release_locks(const char *path, const struct stat *sb,
			 int typeflags, struct FTW *ftw)
{
	size_t len;
	char tmp_path[PATH_MAX];

	strncpy(tmp_path, path, PATH_MAX);
	len = strlen(tmp_path);
	if (strcmp(&tmp_path[len-3], ".BE"))
		return 0;
	/* strip the .PG, ods_lock_cleanup will append it */
	tmp_path[len-3] = '\0';
	ods_lock_cleanup(tmp_path);
	return 0;
}

/**
 * \brief Print container lock information
 *
 * Prints information about the locks held on the container to a FILE pointer.
 *
 * \param sos	The container handle
 * \param fp	The FILE pointer
 */
int sos_container_lock_info(const char *path, FILE *fp)
{
	int rc;
	rc = nftw(path, show_locks, 1024, FTW_DEPTH);
	return rc;
}

/**
 * \brief Release locks held by dead processes
 *
 * It is possible for a process to die while holding container
 * locks. This renders the container inaccessible. This function will
 * release locks held by dead processes.
 *
 * \param sos	The container handle
 */
int sos_container_lock_cleanup(const char *path)
{
	int rc;

	rc = nftw(path, release_locks, 1024, FTW_DEPTH | FTW_PHYS);
	return rc;
}

void sos_inuse_obj_info(sos_t sos, FILE *outp)
{
	sos_obj_t obj;
	if (!outp)
		outp = stdout;
	fprintf(outp, "Active Objects\n");
	fprintf(outp, "%-10s %-12s %s\n", "Ref Count", "Obj Ref", "Schema");
	fprintf(outp, "---------- ------------ ------------\n");
	LIST_FOREACH(obj, &sos->obj_list, entry) {
		sos_obj_ref_str_t ref_str;
		fprintf(outp, "%10d %-12s %s\n",
			obj->ref_count, sos_obj_ref_to_str(obj->obj_ref, ref_str), obj->schema->data->name);
	}
	fprintf(outp, "---------- ------------ ------------\n");
}

void sos_free_obj_info(sos_t sos, FILE *outp)
{
	sos_obj_t obj;
	if (!outp)
		outp = stdout;
	fprintf(outp, "Free Objects\n");
	fprintf(outp, "%-10s %-12s %s\n", "Ref Count", "Obj Ref", "Schema");
	fprintf(outp, "---------- ------------ ------------\n");
	LIST_FOREACH(obj, &sos->obj_free_list, entry) {
		sos_obj_ref_str_t ref_str;
		fprintf(outp, "%10d %-12s %s\n",
			obj->ref_count, sos_obj_ref_to_str(obj->obj_ref, ref_str), obj->schema->data->name);
	}
	fprintf(outp, "---------- ------------ ------------\n");
}

static void free_sos(sos_t sos, sos_commit_t flags)
{
	struct ods_rbn *rbn;
	sos_obj_t obj;
	int inuse_count = 0;

	/* There should be no objects on the active list */
	if (!LIST_EMPTY(&sos->obj_list)) {
		LIST_FOREACH(obj, &sos->obj_list, entry) {
			inuse_count += 1;
		}
	}
	if (inuse_count) {
		sos_error("Leaking %d objects that were inuse when container '%s' was closed\n",
			  inuse_count, sos->path);
		return;
	}

	/* Iterate through the object free list and free all the objects */
	while (!LIST_EMPTY(&sos->obj_free_list)) {
		obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(obj, entry);
		free(obj);
	}

	/* Iterate through all the schema and free each one */
	while (NULL != (rbn = ods_rbt_min(&sos->schema_name_rbt))) {
		ods_rbt_del(&sos->schema_name_rbt, rbn);
		__sos_schema_free(container_of(rbn, struct sos_schema_s, name_rbn));
	}
	if (sos->path)
		free(sos->path);
	if (sos->schema_idx)
		ods_idx_close(sos->schema_idx, flags);
	if (sos->schema_ods)
		ods_close(sos->schema_ods, flags);
	if (sos->idx_idx)
		ods_idx_close(sos->idx_idx, flags);
	if (sos->idx_udata)
		ods_obj_put(sos->idx_udata);
	if (sos->idx_ods)
		ods_close(sos->idx_ods, flags);
	sos_part_t part;
	if (sos->primary_part)
		sos_ref_put(&sos->primary_part->ref_count, "primary_part");
	while (!TAILQ_EMPTY(&sos->part_list)) {
		part = TAILQ_FIRST(&sos->part_list);
		TAILQ_REMOVE(&sos->part_list, part, entry);
		sos_ref_put(&part->ref_count, "part_list");
	}
	if (sos->part_ref_udata)
		ods_obj_put(sos->part_ref_udata);
	if (sos->part_ref_ods)
		ods_close(sos->part_ref_ods, flags);
	pthread_mutex_destroy(&sos->lock);
	free(sos);
}

int64_t __sos_schema_name_cmp(void *a, const void *b, void *arg)
{
	return strcmp((char *)a, (char *)b);
}

int64_t __sos_schema_id_cmp(void *a, const void *b, void *arg)
{
	return uuid_compare(a, b);
}

struct sos_version_s sos_container_version(sos_t sos)
{
	struct sos_version_s vers;
	struct ods_version_s overs = ods_version(sos->schema_ods);
	vers.major = overs.major;
	vers.minor = overs.minor;
	vers.fix = overs.fix;
	memcpy(vers.git_commit_id, overs.commit_id, sizeof(overs.commit_id));
	return vers;
}

static int is_supported_version(uint64_t v1, uint64_t v2)
{
	if ((v1 & SOS_VERSION_MASK) == (v2 & SOS_VERSION_MASK))
		return 1;
	return 0;
}

/**
 * \brief Open a Container
 *
 * Open a SOS container. If successfull, the <tt>c</tt> parameter will
 * contain a valid \c sos_t handle on exit.
 *
 * \param path_arg	Pathname to the Container. If SOS_PERM_CREAT is
 *			specified all sub-directories in the path will be
 *			create.
 * \param o_perm	The requested access permissions as follows:
 *	SOS_PERM_RD	Read-only access
 *	SOS_PERM_WR	Write-only access
 *	SOS_PERM_RW	Read-write access
 *	SOS_PERM_CREAT	Create the container if it does not already exist.
 *			If this flag is specified, the \c o_mode parameter
 *			must immediately follow with the desired file
 *			permission bits. See the open() system call.
 *	SOS_PERM_USER	Open the container as a specific user/group. This
 *			flag cannot be used with the SOS_PERM_CREAT flag.
 *			If specified, the \c o_uid, and \c o_gid parameters
 *			must follow the \c o_perm parameter. This flag is
 *			useful for limiting object visibility and cannot be
 *			used with the SOS_PERM_CREAT flag.
 *	SOS_BE_MMAP	Use the Memory Mapped Object Store back-end
 *	SOS_BE_LSOS	Use the Log Structured Obect Store back end
 * \param o_mode	If \c o_perm contains the SOS_PERM_CREAT flag, this option
 *			immediately follows the  \c o_perm optino. See the open()
 *			system call.
 * \param user		if \c o_perm contains the SOS_PERM_USER flag, the \c user
 *			parameter follows either the \c o_perm parameter or the
 *			\c o_mode parameter if SOS_PERM_CREAT is specified.
 * \param group		if \c o_perm contains the SOS_PERM_USER flag, the \c group
 *			parameter follows the \c user parameter.
 * \retval !NULL	The sos_t handle for the container.
 * \retval NULL		An error occured, consult errno for the reason.
 */
sos_t sos_container_open(const char *path_arg, sos_perm_t o_perm, ...)
{
	char tmp_path[PATH_MAX];
	char *path = NULL;
	sos_t sos;
	struct stat sb;
	ods_iter_t iter = NULL;
	int rc;
	int lock_fd;
	int o_mode;
	int need_part = 0;
	va_list ap;
	uid_t euid = geteuid();
	gid_t egid = getegid();

	va_start(ap, o_perm);
	if (o_perm & SOS_PERM_CREAT) {
		o_mode = va_arg(ap, int);
	} else {
		o_mode = 0666;
	}
	if (o_perm & SOS_PERM_USER) {
		euid = va_arg(ap, int);
		egid = va_arg(ap, int);
	}
	va_end(ap);

	if (stat(path_arg, &sb) < 0) {
		if (0 == (o_perm & SOS_PERM_CREAT)) {
			return NULL;
		}
	}

	(void)sos_container_lock_cleanup(path_arg);

	if (strlen(path_arg) >= SOS_PART_PATH_LEN) {
		errno = E2BIG;
		return NULL;
	}
	if (path_arg[0] != '/') {
		if (!getcwd(tmp_path, sizeof(tmp_path)))
			return NULL;
		if (strlen(tmp_path) + strlen(path_arg) > SOS_PART_PATH_LEN) {
			errno = E2BIG;
			return NULL;
		}
		strcat(tmp_path, "/");
		strcat(tmp_path, path_arg);
		path = strdup(tmp_path);
	} else
		path = strdup(path_arg);
	if (!path) {
		errno = ENOMEM;
		return NULL;
	}
	sos = calloc(1, sizeof(*sos));
	if (!sos) {
		errno = ENOMEM;
		free(path);
		return NULL;
	}
	pthread_mutex_init(&sos->lock, NULL);
	LIST_INIT(&sos->obj_list);
	LIST_INIT(&sos->obj_free_list);
	TAILQ_INIT(&sos->part_list);

	/* Take the SOS file lock */
	char *dir = strdup(path);
	if (!dir)
		return NULL;
	char *base = strdup(path);
	if (!base)
		return NULL;
	sprintf(tmp_path, "%s/.%s.lock", dirname(dir), basename(base));
	free(dir);
	free(base);
	lock_fd = open(tmp_path, O_RDWR | O_CREAT, 0666);
	if (lock_fd < 0)
		return NULL;
	rc = flock(lock_fd, LOCK_EX);
	if (rc) {
		close(lock_fd);
		errno = rc;
		return NULL;
	}

	/* Stat the container path to get the file mode bits */
	sos->path = path;
	rc = stat(sos->path, &sb);
	if (rc) {
		/* Check if this container exists */
		if (errno != ENOENT)
			goto err;
		rc = __sos_container_new(path_arg, &o_perm, o_mode);
		if (rc)
			goto err;
		need_part = 1;
		rc = stat(sos->path, &sb);
		if (rc)
			goto err;
		o_perm &= ~SOS_PERM_CREAT;
	} else {
		o_mode = sb.st_mode & 0666;
	}

	sos->o_mode = o_mode;
	sos->o_perm = (ods_perm_t)o_perm;
	ods_rbt_init(&sos->schema_name_rbt, __sos_schema_name_cmp, NULL);
	ods_rbt_init(&sos->schema_id_rbt, __sos_schema_id_cmp, NULL);
	sos->schema_count = 0;

	rc = __sos_config_init(sos);
	if (rc) {
		sos_error("Error %d initializing configuration on open of %s\n", errno, path_arg);
		goto err;
	}
	char *be_type = sos_container_config_get(path, "BACKEND_TYPE");
	if (be_type) {
		if (0 == strcasecmp(be_type, "LSOS"))
			sos->o_perm |= SOS_BE_LSOS;
		else
			sos->o_perm |= SOS_BE_MMOS;
		free(be_type);
	} else {
		sos->o_perm |= SOS_BE_MMOS;
	}

	/* Open the ODS containing the Index objects */
	sprintf(tmp_path, "%s/.__index", path);
	sos->idx_ods = ods_open(tmp_path, sos->o_perm, sos->o_mode);
	if (!sos->idx_ods)
		goto err;

	/* Get the index object user data */
	sos->idx_udata = ods_get_user_data(sos->idx_ods);
	if (SOS_IDXDIR_UDATA(sos->idx_udata)->signature != SOS_IDXDIR_SIGNATURE) {
		errno = EINVAL;
		sos_error("Database %s index directory is not valid. "
			  "Expected a signature of %llx, got %llx\n",
			  path_arg,
			  SOS_IDXDIR_SIGNATURE,
			  SOS_IDXDIR_UDATA(sos->idx_udata)->signature);
		ods_obj_put(sos->idx_udata);
		goto err;
	}

	/* Open the index of index objects */
	sprintf(tmp_path, "%s/.__index_idx", path);
	sos->idx_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->idx_idx) {
		sos_error("Error %d opening index index in open of %s\n", errno, path_arg);
		ods_obj_put(sos->idx_udata);
		goto err;
	}

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/.__schemas", path);
	sos->schema_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->schema_ods) {
		sos_error("Error %d opening schema ODS in open of %s\n", errno, path_arg);
		goto err;
	}
	ods_obj_t udata = ods_get_user_data(sos->schema_ods);
	if ((SOS_SCHEMA_UDATA(udata)->signature != SOS_SCHEMA_SIGNATURE)) {
		errno = EINVAL;
		sos_error("Schema ODS in %s is corrupted expected %llX, got %llx\n", path_arg,
			  SOS_SCHEMA_SIGNATURE,
			  SOS_SCHEMA_UDATA(udata)->signature);
		ods_obj_put(udata);
		goto err;
	}

	if (!is_supported_version(SOS_SCHEMA_UDATA(udata)->version, SOS_LATEST_VERSION)) {
		errno = EPROTO;
		sos_error("Schema ODS in %s is an unsupported version expected %llX, got %llx\n",
			  path_arg,
			  SOS_LATEST_VERSION,
			  SOS_SCHEMA_UDATA(udata)->version);
		ods_obj_put(udata);
		goto err;
	}
	ods_obj_put(udata);

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/.__schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->schema_idx)
		goto err;

	/*
	 * Build the schema dictionary
	 */
	iter = ods_iter_new(sos->schema_idx);
	ods_lock(sos->schema_ods, 0, NULL);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		sos_obj_ref_t obj_ref;
		obj_ref.idx_data = ods_iter_data(iter);
		ods_obj_t schema_obj = ods_ref_as_obj(sos->schema_ods, obj_ref.ref.obj);
		sos_schema_t schema = __sos_schema_init(sos, schema_obj);
		if (!schema) {
			ods_unlock(sos->schema_ods, 0);
			goto err;
		}
	}
	ods_unlock(sos->schema_ods, 0);

	/*
	 * Open the partitions
	 */
	int acc = 0;
	if (o_perm & SOS_PERM_RD)
		acc = 06;
	if (o_perm & SOS_PERM_WR)
		acc |= 04;
	/* Open the partition ODS */
	sprintf(tmp_path, "%s/.__part", sos->path);
	sos->part_ref_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_ref_ods)
		goto err;
	sos->part_ref_udata = ods_get_user_data(sos->part_ref_ods);
	if (!sos->part_ref_udata)
		goto err;
	if (need_part) {
		snprintf(tmp_path, sizeof(tmp_path), "%s/default", path_arg);
		rc = __sos_part_create(tmp_path, "default container partition", sos->o_perm, o_mode);
		if (rc)
			goto err;
		rc = sos_part_attach(sos, "default", tmp_path);
		if (rc)
			goto err;
		sos_part_t part = sos_part_by_name(sos, "default");
		if (!part)
			goto err;
		rc = sos_part_chown(part, euid, egid);
		if (rc)
			goto err;
		rc = sos_part_chmod(part, o_mode);
		if (rc)
			goto err;
		rc = sos_part_state_set(part, SOS_PART_STATE_PRIMARY);
		sos_part_put(part);
		if (rc)
			goto err;
	} else {
		rc = __sos_open_partitions(sos, tmp_path, euid, egid, acc);
		if (rc) {
			errno = rc;
			goto err;
		}
	}
	pthread_mutex_lock(&cont_list_lock);
	LIST_INSERT_HEAD(&cont_list, sos, entry);
	pthread_mutex_unlock(&cont_list_lock);

	ods_iter_delete(iter);
	close(lock_fd);
	ods_commit(sos->part_ref_ods, ODS_COMMIT_SYNC);
	ods_commit(sos->schema_ods, ODS_COMMIT_SYNC);
	ods_commit(sos->idx_ods, ODS_COMMIT_SYNC);
	return sos;
err:
	if (iter)
		ods_iter_delete(iter);
	free_sos(sos, SOS_COMMIT_ASYNC);
	close(lock_fd);
	return NULL;
}

/**
 * \brief Verify a container
 *
 * Perform internal consistency checks on the container. Use this to
 * verify that the container is not corrrupted.
 *
 * \param sos The container handle
 * \returns 0 if the database is healthy
 */
int sos_container_verify(sos_t sos)
{
	char path[PATH_MAX];
	sos_obj_ref_t idx_ref;
	ods_idx_t idx;
	int rc, res = 0;
	ods_iter_t iter = ods_iter_new(sos->idx_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		idx_ref.idx_data = ods_iter_data(iter);
		ods_obj_t idx_obj = ods_ref_as_obj(sos->idx_ods, idx_ref.ref.obj);
		fprintf(stdout, "Verifying %s\n", SOS_IDX(idx_obj)->name);
		snprintf(path, PATH_MAX, "%s/%s_idx", sos->path, SOS_IDX(idx_obj)->name);
		idx = ods_idx_open(path, sos->o_perm);
		ods_obj_put(idx_obj);
		if (!idx) {
			sos_error("Error %d opening %s\n", errno, path);
			goto out;
		}
		res = ods_idx_verify(idx, stdout);
	}
out:
	ods_iter_delete(iter);
	return res;
}

/**
 * \brief Return information about a container
 *
 * Fills a Unix struct stat buffer with information about a container's meta data.
 *
 * \param sos The container handle
 * \param sb The struct stat buffer
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
int sos_container_stat(sos_t sos, struct stat *sb)
{
	sos_part_t part = TAILQ_FIRST(&sos->part_list);
	return ods_stat(part->obj_ods, sb);
}

/**
 * \brief Close a Container
 *
 * This function commits changes to stable storage and releases all
 * in-memory resources associated with the container.
 *
 * If SOS_COMMIT_SYNC is specified in the flags parameter, the
 * function will wait until the changes are commited before returning.
 *
 * \param sos	The SOS container handle
 * \param flags	The commit flags
 */
void sos_container_close(sos_t sos, sos_commit_t flags)
{
	pthread_mutex_lock(&cont_list_lock);
	LIST_REMOVE(sos, entry);
	pthread_mutex_unlock(&cont_list_lock);

	free_sos(sos, SOS_COMMIT_ASYNC);
}

/**
 * @brief Clone a thin copy of the container
 *
 * Create a new container with all of the schema from the container
 *
 * @param sos The container handle
 * @param path The location of the clone
 * @retval EINUSE The \c path is in use
 * @retval ENOMEM The filesystem is full
 */
int sos_container_clone(sos_t sos, const char *path)
{
	int rc = 0;
	sos_t clone;

	/* Make certain the destination does not already exist */
	clone = sos_container_open(path, SOS_PERM_RW, sos->o_mode);
	if (clone)
		return EEXIST;

	clone = sos_container_open(path, SOS_PERM_RW | SOS_PERM_CREAT, sos->o_mode);
	if (!clone)
		return errno;

	sos_schema_t schema;
	for (schema = sos_schema_first(sos); schema; schema = sos_schema_next(schema)) {
		sos_schema_t copy = sos_schema_dup(schema);
		if (!copy) {
			rc = errno;
			goto out;
		}
		rc = sos_schema_add(clone, copy);
		if (rc)
			goto out;
	}
out:
	sos_container_close(clone, SOS_COMMIT_ASYNC);
	return rc;
}

/*
 * This function steals the reference count of the input ods_obj
 */
sos_obj_t __sos_init_obj_no_lock(sos_t sos, sos_schema_t schema, sos_part_t part, ods_obj_t ods_obj,
				 sos_obj_ref_t obj_ref)
{
	sos_obj_t sos_obj;

	/* Verify the reference provided */
	if (sos) {
		if (ods_obj->ods && !ods_ref_valid(ods_obj->ods, obj_ref.ref.obj)) {
			sos_obj_ref_str_t ref_str;
			sos_error("Invalid object reference %s",
				sos_obj_ref_to_str(obj_ref, ref_str));
			return NULL;
		}
		if (!LIST_EMPTY(&sos->obj_free_list)) {
			sos_obj = LIST_FIRST(&sos->obj_free_list);
			LIST_REMOVE(sos_obj, entry);
		} else {
			sos_obj = malloc(sizeof *sos_obj);
		}
	} else {
		sos_obj = malloc(sizeof *sos_obj);
	}
	if (!sos_obj)
		return NULL;
	uuid_copy(SOS_OBJ(ods_obj)->schema_uuid, schema->data->uuid);
	sos_obj->sos = sos;
	if (part)
		sos_obj->part = sos_part_get(part);
	else
		part = NULL;
	sos_obj->obj = ods_obj;
	sos_obj->obj_ref = obj_ref;
	ods_atomic_inc(&schema->data->ref_count);
	sos_obj->schema = schema;
	sos_obj->ref_count = 1;
	size_t array_size = __sos_obj_array_size(sos_obj);
	sos_obj->next_array_off = schema->data->obj_sz + array_size;
	sos_obj->size = sos_obj->next_array_off;
	if (sos)
		LIST_INSERT_HEAD(&sos->obj_list, sos_obj, entry);
	return sos_obj;
}

sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, sos_part_t part, ods_obj_t ods_obj,
			 sos_obj_ref_t obj_ref)
{
	sos_obj_t sos_obj;
	if (sos)
		pthread_mutex_lock(&sos->lock);
	sos_obj = __sos_init_obj_no_lock(sos, schema, part, ods_obj, obj_ref);
	if (sos)
		pthread_mutex_unlock(&sos->lock);
	return sos_obj;
}

/**
 * \brief Allocate an object from the SOS object store.
 *
 * This call will automatically extend the size of the backing store
 * to accomodate the new object. This call will fail if there is
 * insufficient disk space. Use the sos_obj_index() to add the object
 * to all indices defined by its object class.
 *
 * See the sos_schema_by_name() function call for information on how to
 * obtain a schema handle.
 *
 * \param schema	The schema handle
 * \returns Pointer to the new object
 * \returns NULL if there is an error
 */
#define ARRAY_RESERVE 256

sos_obj_t sos_obj_new(sos_schema_t schema)
{
	ods_obj_t ods_obj;
	sos_obj_t sos_obj;
	sos_part_t part;
	sos_obj_ref_t obj_ref;

	if (!schema || !schema->sos) {
		errno = EINVAL;
		return NULL;
	}
	part = __sos_primary_obj_part(schema->sos);
	if (!part) {
		errno = ENOSPC;
		return NULL;
	}
	size_t array_data_sz = schema->data->array_cnt * ARRAY_RESERVE;
	ods_obj = ods_obj_malloc(schema->data->obj_sz + array_data_sz);
	if (!ods_obj)
		goto err_0;
	memset(ods_obj->as.ptr, 0, schema->data->obj_sz);
	uuid_copy(obj_ref.ref.part_uuid, SOS_PART_UDATA(part->udata_obj)->uuid);
	obj_ref.ref.obj = ods_obj_ref(ods_obj);
	sos_obj = __sos_init_obj(schema->sos, schema, NULL, ods_obj, obj_ref);
	if (!sos_obj)
		goto err_1;
	/* Initialize array attributes to zero length */
	__sos_init_array_values(schema, sos_obj);
	return sos_obj;
err_1:
	ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
err_0:
	return NULL;
}

/**
 * @brief Create a new object and populate it with data
 *
 * The \c data provided is copied into the object following the object
 * header. Note that object data is variable length when there are
 * arrays in the schema.
 *
 * @param schema The object schema
 * @param data The attribute data
 * @param data_size The data size in bytes
 * @return sos_obj_t The object handle
 */
sos_obj_t sos_obj_new_with_data(sos_schema_t schema, uint8_t *data, size_t data_size)
{
	ods_obj_t ods_obj;
	sos_obj_t sos_obj;
	sos_part_t part = NULL;
	sos_obj_ref_t obj_ref;

	if (!schema)
	{
		errno = EINVAL;
		return NULL;
	}
	if (schema->sos) {
		part = __sos_primary_obj_part(schema->sos);
		if (!part)
		{
			errno = ENOSPC;
			return NULL;
		}
	}
	if (data_size < (schema->data->obj_sz - sizeof(struct sos_obj_data_s))) {
		errno = EINVAL;
		return NULL;
	}
	ods_obj = ods_obj_malloc(data_size + sizeof(struct sos_obj_data_s));
	if (!ods_obj)
		goto err_0;
	if (part)
		uuid_copy(obj_ref.ref.part_uuid, SOS_PART_UDATA(part->udata_obj)->uuid);
	obj_ref.ref.obj = ods_obj_ref(ods_obj);
 	memcpy(&ods_obj->as.bytes[sizeof(struct sos_obj_data_s)], data, data_size);
	sos_obj = __sos_init_obj(schema->sos, schema, NULL, ods_obj, obj_ref);
	if (!sos_obj)
		goto err_1;
	return sos_obj;
err_1:
	if (schema->sos)
		ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
err_0:
	return NULL;
}

/**
 * \brief Allocate a SOS object in memory
 *
 * This call allocates a memory based object that is not stored in
 * the container.
 *
 * \param schema        The schema handle
 * \returns Pointer to the new object
 * \returns NULL if there is an error
 */
sos_obj_t sos_obj_malloc(sos_schema_t schema)
{
	ods_obj_t ods_obj;
	sos_obj_ref_t obj_ref = NULL_REF;
	if (!schema)
	{
		errno = EINVAL;
		return NULL;
	}
	size_t array_data_sz = schema->data->array_cnt * ARRAY_RESERVE;
	ods_obj = ods_obj_malloc(schema->data->obj_sz + array_data_sz);
	if (!ods_obj)
		goto err_0;
	memset(ods_obj->as.ptr, 0, schema->data->obj_sz + array_data_sz);
	return __sos_init_obj(NULL, schema, NULL, ods_obj, obj_ref);
err_0:
	return NULL;
}

/**
 * \brief Copy the data in one object to another
 *
 * Copy all of the data from the object specified by \c src_obj
 * to the object specified by \c dst_obj. The objects do not
 * need to be in the same container. If \c src_obj contains
 * array attributes, an array of the required size is allocated
 * in the destination object and the data copied.
 *
 * The objects do not need to have the same schema, however, the
 * type of the attributes in \c src_obj must match the types of
 * the attributes in the same order in \c dst_obj.
 *
 * \param dst_obj	The object to copy to
 * \param src_obj	The object to copy from
 * \retval 0		The object data was sucessfully copied
 * \retval EINVAL	The source and destination object attributes object attributes don't match
 * \retval ENOMEM	There were insufficient resources to assign the object
 */
int sos_obj_copy(sos_obj_t dst_obj, sos_obj_t src_obj)
{
	sos_attr_t src_attr, dst_attr;
	sos_value_data_t src_data, dst_data;

	dst_attr = TAILQ_FIRST(&dst_obj->schema->attr_list);
	TAILQ_FOREACH(src_attr, &src_obj->schema->attr_list, entry) {
		if (sos_attr_type(src_attr) != sos_attr_type(dst_attr))
			return EINVAL;
		if (sos_attr_type(src_attr) == SOS_TYPE_JOIN) {
			/* JOIN attrs do not occupy space in the object data */
			dst_attr = TAILQ_NEXT(dst_attr, entry);
			continue;
		}
		if (!sos_attr_is_array(src_attr)) {
			src_data = sos_obj_attr_data(src_obj, src_attr, NULL);
			dst_data = sos_obj_attr_data(dst_obj, dst_attr, NULL);
			if (!src_data || !dst_data)
				return EINVAL;
			memcpy(dst_data->prim.struc_, src_data->prim.struc_,
				sos_attr_size(src_attr));
		} else {
			struct sos_value_s src_v_, dst_v_;
			sos_value_t src_value;
			sos_value_t dst_value;
			src_value = sos_value_init(&src_v_, src_obj, src_attr);
			if (!src_value)
				return EINVAL;
			dst_value = sos_value_init(&dst_v_, dst_obj, dst_attr);
			if (!dst_value)
				return ENOMEM;
			memcpy(sos_array(dst_value), sos_array(src_value),
			   sos_value_size(dst_value));
			sos_value_put(src_value);
			sos_value_put(dst_value);
		}
		dst_attr = TAILQ_NEXT(dst_attr, entry);
	}
	return 0;
}

int sos_obj_attr_copy(sos_obj_t dst_obj, sos_attr_t dst_attr,
		      sos_obj_t src_obj, sos_attr_t src_attr)
{
	sos_value_data_t src_data, dst_data;

	if (sos_attr_type(src_attr) != sos_attr_type(dst_attr))
		return EINVAL;
	if (sos_attr_type(src_attr) == SOS_TYPE_JOIN)
		return 0;

	if (!sos_attr_is_array(src_attr)) {
		src_data = sos_obj_attr_data(src_obj, src_attr, NULL);
		dst_data = sos_obj_attr_data(dst_obj, dst_attr, NULL);
		if (!src_data || !dst_data)
			return EINVAL;
		memcpy(dst_data->prim.struc_, src_data->prim.struc_,
		       sos_attr_size(src_attr));
	} else {
		struct sos_value_s src_v_, dst_v_;
		sos_value_t src_value;
		sos_value_t dst_value;
		src_value = sos_value_init(&src_v_, src_obj, src_attr);
		if (!src_value)
			return EINVAL;
		dst_value = sos_array_new(&dst_v_, dst_attr, dst_obj,
					  sos_array_count(src_value));
		if (!dst_value)
			return ENOMEM;
		sos_array(dst_value)->count = sos_array(src_value)->count;
		memcpy(sos_array_data(dst_value, byte_),
		       sos_array_data(src_value, byte_),
		       sos_value_size(src_value));
		sos_value_put(src_value);
		sos_value_put(dst_value);
	}
	return 0;
}

sos_obj_ref_t sos_obj_ref(sos_obj_t obj)
{
	if (!obj)
		return NULL_REF;
	return obj->obj_ref;
}

static int ref_is_null(sos_obj_ref_t ref)
{
	return (uuid_is_null(ref.ref.part_uuid) && (ref.ref.obj == 0));
}

/**
 * \brief  Return the object associated with the reference
 *
 * This function will return a sos_obj_t for the object that is referred
 * to by 'ref'. Use the function sos_obj_from_value() to obtain an
 * object from a sos_value_t value.
 *
 * \param sos The container handle
 * \param ref The object reference
 * \retval The object to which the reference refers.
 * \retval NULL The reference did not point to a well formed object, or the schema
 *              in the object header was not part of the container.
 */
sos_obj_t sos_ref_as_obj(sos_t sos, sos_obj_ref_t ref)
{
	ods_obj_t ods_obj;
	if (ref_is_null(ref))
		return NULL;

	sos_part_t part = __sos_part_find_by_uuid(sos, ref.ref.part_uuid);
	if (!part) {
		errno = ENOENT;
		return NULL;
	}
	ods_obj = ods_ref_as_obj(part->obj_ods, ref.ref.obj);
	if (!ods_obj) {
		errno = ENOENT;
		return NULL;
	}

	/* Get the schema id from the SOS object */
	sos_obj_data_t sos_obj = ods_obj->as.ptr;
	sos_schema_t schema = sos_schema_by_uuid(sos, sos_obj->schema_uuid);
	if (!schema)
		return NULL;

	return __sos_init_obj(sos, schema, part, ods_obj, ref);
}

/**
 * \brief Returns an Object's schema
 * \param obj The object handle
 * \retval The schema used to create the object.
 */
sos_schema_t sos_obj_schema(sos_obj_t obj)
{
	return obj->schema;
}

/**
 * \brief Release the storage consumed by the object in the SOS object store.
 *
 * Deletes the object and any arrays to which the object refers. It
 * does not delete an object that is referred to by this object,
 * i.e. SOS_TYPE_OBJ_REF attribute values. This function does does not
 * remove the object from indices that may refer to the object.
 *
 * This function does not drop any references on the memory resources
 * for this object. Object references must still be dropped with the
 * sos_obj_put() function.
 *
 * \param obj	Pointer to the object
 */
void sos_obj_delete(sos_obj_t obj)
{
	if (!obj->obj->ods) {
		free(obj->obj);
	} else {
		ods_obj_delete(obj->obj);
		ods_obj_put(obj->obj);
		obj->obj = NULL;
	}
}

/**
 * \brief Take a reference on an object
 *
 * SOS objects are reference counted. This function takes a reference
 * on a SOS object and returns a pointer to the object. The typical
 * calling sequence is:
 *
 *     sos_obj_t my_obj_ptr = sos_obj_get(obj);
 *
 * This allows for the object to be safely pointed to from multiple
 * places. The sos_obj_put() function is used to drop a reference on
 * a SOS object. For example:
 *
 *     sos_obj_put(my_obj_ptr);
 *     my_obj_ptr = NULL;
 *
 * \param obj	The SOS object handle
 * \retval The object handle
 */
sos_obj_t sos_obj_get(sos_obj_t obj)
{
	ods_atomic_inc(&obj->ref_count);
	return obj;
}

/**
 * \brief Drop a reference on an object
 *
 * SOS objects are reference counted. The memory consumed by the
 * object is not released until all references have been dropped. This
 * refers only to references in main memory. The object will continue
 * to exist in persistent storage. See the sos_obj_delete() function
 * for information on removing an object from persistent storage.
 *
 * \param obj	The object handle. If NULL, this function is a no-op.
 */
void sos_obj_put(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		if (!sos) {
			if (obj->obj)
				ods_obj_put(obj->obj);
			free(obj);
			return;
		}
		ods_obj_put(obj->obj);
		sos_part_put(obj->part);
		pthread_mutex_lock(&sos->lock);
		LIST_REMOVE(obj, entry);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
		pthread_mutex_unlock(&sos->lock);
	}
}
void __sos_obj_put_no_lock(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		if (!sos)
			return;
		ods_obj_put(obj->obj);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
	}
}

/*
 * Find the ODS index from the partition in which an object
 * is allocated.
 */
ods_idx_t __sos_idx_find(sos_index_t index, sos_obj_t obj)
{
	ods_idx_ref_t iref;
	sos_part_t part = obj->part;
	if (part) {
		LIST_FOREACH(iref, &index->active_idx_list, entry) {
			if (iref->part == part)
				return iref->idx;
		}
	}
	return NULL;
}

/**
 * \brief Remove an object from the SOS
 *
 * This removes an object from all indexes of which it is a
 * member. The object itself is not destroyed. Use the
 * sos_obj_delete() function to release the storage consumed by the
 * object itself.
 *
 * \param obj	Handle for the object to remove
 *
 * \returns 0 on success.
 * \returns Error code on error.
 */
int sos_obj_remove(sos_obj_t obj)
{
	sos_attr_t attr;
	size_t key_sz;
	int rc;
	sos_key_t key;
	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		sos_index_t index;
		struct sos_value_s v_;
		sos_value_t value;
		index = sos_attr_index(attr);
		if (!index)
			continue;
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		key = sos_key_new(key_sz);
		if (!key) {
			sos_value_put(value);
			return ENOMEM;
		}
		ods_key_set(key, sos_value_as_key(value), key_sz);
		ods_idx_t idx = __sos_idx_find(index, obj);
		if (idx) {
			rc = ods_idx_delete(idx, key, &obj->obj_ref.idx_data);
		} else {
			rc = ENOENT;
			sos_error("The ODS idx for the object partition is missing.");
		}
		sos_key_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

/**
 * \brief Add an object to its indexes
 *
 * Add an object to all the indices defined in its schema. This
 * function should only be called after all attributes that have
 * indexes have had their values set.
 *
 * \param obj	Handle for the object to add
 *
 * \retval 0	Success
 * \retval -1	An error occurred. Refer to errno for detail.
 */
int sos_obj_index(sos_obj_t obj)
{
	struct sos_value_s v_;
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	sos_key_t the_key = NULL;
	SOS_KEY(key);
	int rc;

	if (NULL == obj->obj->ods) {
		rc = sos_obj_commit(obj);
		if (rc)
			return rc;
	}

	TAILQ_FOREACH(attr, &obj->schema->idx_attr_list, idx_entry) {
		sos_index_t index = sos_attr_index(attr);
		if (!index)
			return errno;
		value = sos_value_init(&v_, obj, attr);
		if (!value) {
			/* Array value not set, skip */
			continue;
		}
		key_sz = sos_value_size(value);
		if (key_sz < 254) {
			the_key = key;
		} else {
			the_key = sos_key_new(key_sz);
		}
		sos_key_set(the_key, sos_value_as_key(value), key_sz);
		rc = sos_index_insert(index, the_key, obj);
		if (the_key != key)
			sos_key_put(the_key);
		sos_value_put(value);
		if (rc)
			goto err;
	}
	return 0;
err:
	return rc;
}

/**
 * \brief Set an object attribute's value from a string
 *
 * This convenience function set's an object's attribute value specified as a
 * string. The attribute to set is specified by name.
 *
 * For example:
 *
 *     int rc = sos_obj_attr_by_name_from_str(an_obj, "my_int_attr", "1234");
 *     if (!rc)
 *        printf("Success!!\n");
 *
 * See the sos_obj_attr_from_str() function to set the value with a string if
 * the attribute handle is known.
 *
 * \param sos_obj	The object handle
 * \param attr_name	The attribute name
 * \param attr_value	The attribute value as a string
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 Success
 * \retval EINVAL The string format was invalid for the attribute type
 * \retval ENOSYS There is no string formatter for this attribute type
 */
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr)
{
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return ENOENT;

	return sos_obj_attr_from_str(sos_obj, attr, attr_value, endptr);
}

char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj,
				  const char *attr_name, char *str, size_t len)
{
	sos_attr_t attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return NULL;
	return sos_obj_attr_to_str(sos_obj, attr, str, len);
}

char *sos_obj_attr_by_id_to_str(sos_obj_t sos_obj,
				int attr_id, char *str, size_t len)
{
	sos_attr_t attr = sos_schema_attr_by_id(sos_obj->schema, attr_id);
	if (!attr)
		return NULL;
	return sos_obj_attr_to_str(sos_obj, attr, str, len);
}

static void __attribute__ ((constructor)) sos_lib_init(void)
{
	LIST_INIT(&cont_list);
	pthread_mutex_init(&cont_list_lock, NULL);
}

static void __attribute__ ((destructor)) sos_lib_term(void)
{
}

/** @} */

