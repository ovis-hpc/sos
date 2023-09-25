# Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
# Copyright (c) 2019 NTESS Corporation. All rights reserved.
# Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the U.S. Government.
# Export of this program may require a license from the United States
# Government.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the BSD-type
# license below:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#      Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#      Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#      Neither the name of NTESS Corporation, Open Grid Computing nor
#      the names of any contributors may be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
#
#      Modified source versions must be plainly marked as such, and
#      must not be misrepresented as being the original software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cython
cimport numpy as np
from libc.stdint cimport *
from libc.string cimport *

cdef extern from "errno.h":
    cdef int errno

    cdef enum:
        E2BIG,
        EACCES,
        EADDRINUSE,
        EADDRNOTAVAIL,
        EAFNOSUPPORT,
        EAGAIN,
        EALREADY,
        EBADE,
        EBADF,
        EBADFD,
        EBADMSG,
        EBADR,
        EBADRQC,
        EBADSLT,
        EBUSY,
        ECHILD,
        ECHRNG,
        ECOMM,
        ECONNABORTED,
        ECONNREFUSED,
        ECONNRESET,
        EDEADLK,
        EDEADLOCK,
        EDESTADDRREQ,
        EDOM,
        EDQUOT,
        EEXIST,
        EFAULT,
        EFBIG,
        EHOSTDOWN,
        EHOSTUNREACH,
        EIDRM,
        EILSEQ,
        EINPROGRESS,
        EINTR,
        EINVAL,
        EIO,
        EISCONN,
        EISDIR,
        EISNAM,
        ELOOP,
        EMFILE,
        EMLINK,
        EMSGSIZE,
        EMULTIHOP,
        ENAMETOOLONG,
        ENETDOWN,
        ENETRESET,
        ENETUNREACH,
        ENFILE,
        ENOBUFS,
        ENODATA,
        ENODEV,
        ENOENT,
        ENOEXEC,
        ENOLCK,
        ENOLINK,
        ENOMEM,
        ENOMSG,
        ENONET,
        ENOPKG,
        ENOPROTOOPT,
        ENOSPC,
        ENOSR,
        ENOSTR,
        ENOSYS,
        ENOTBLK,
        ENOTCONN,
        ENOTDIR,
        ENOTEMPTY,
        ENOTSOCK,
        ENOTSUP,
        ENOTTY,
        ENOTUNIQ,
        ENXIO,
        EOPNOTSUPP,
        EOVERFLOW,
        EPERM,
        EPFNOSUPPORT,
        EPIPE,
        EPROTO,
        EPROTONOSUPPORT,
        EPROTOTYPE,
        ERANGE,
        EREMCHG,
        EREMOTE,
        EREMOTEIO,
        ERESTART,
        EROFS,
        ESHUTDOWN,
        ESPIPE,
        ESOCKTNOSUPPORT,
        ESRCH,
        ESTALE,
        ESTRPIPE,
        ETIME,
        ETIMEDOUT,
        EUSERS,
        EWOULDBLOCK,
        EXFULL

cdef extern from "sys/stat.h":
    cdef struct stat

cdef extern from "sys/time.h":
    ctypedef int time_t
    cdef struct timespec:
        time_t tv_sec
        int64_t tv_nsec

cdef extern from "stdio.h":
    cdef struct _IO_FILE:
        pass
ctypedef _IO_FILE FILE

cdef extern from "unistd.h":
    ctypedef long gid_t
    ctypedef long uid_t

cdef extern from "ods/ods_idx.h":

    cdef struct ods_idx_data_s:
        unsigned char bytes[32]
    ctypedef ods_idx_data_s ods_idx_data_t

    cdef struct ods_timeval_s:
        uint32_t tv_usec
        uint32_t tv_sec

    cdef struct ods_key_value_s:
        uint16_t len
        unsigned char value[0]
        unsigned char byte_[0]
        uint16_t uint16_[0]
        uint32_t uint32_[0]
        uint64_t uint64_[0]
        int16_t int16_[0]
        int32_t int32_[0]
        int64_t int64_[0]
        float float_[0]
        double double_[0]
        long double long_double_[0]
        ods_timeval_s tv_[0]
    ctypedef ods_key_value_s *ods_key_value_t

cdef extern from "config.h":
    cdef char *ODS_COMMIT_ID

cdef extern from "ods/ods.h":
    cdef enum:
        ODS_VER_MAJOR
        ODS_VER_MINOR
        ODS_VER_FIX

    ctypedef uint64_t ods_ref_t
    cdef struct ods_stat:
        timespec st_atim
        timespec st_mtim
        timespec st_ctim
        uint64_t st_size
        uint64_t st_grain_size
        uint64_t st_pg_size
        uint64_t st_pg_count
        uint64_t st_pg_free
        uint64_t st_bkt_count
        uint64_t st_total_blk_free
        uint64_t st_total_blk_alloc
        uint64_t *st_blk_free
        uint64_t *st_blk_alloc
    ctypedef ods_stat *ods_stat_t

    cdef union ods_obj_type_u:
        void *ptr
        int8_t *int8
        uint8_t *uint8
        int16_t *int16
        uint16_t *uint16
        int32_t *int32
        uint32_t *uint32
        int64_t *int64
        uint64_t *uint64
        ods_timeval_s *tv
        char *str
        unsigned char *bytes
        ods_key_value_s *key

    cdef struct ods_obj_s:
        ods_obj_type_u as

cdef extern from "uuid/uuid.h":
    ctypedef unsigned char uuid_t[16];
    void uuid_unparse(uuid_t, char *)
    void uuid_parse(char *, uuid_t)

cdef extern from "sos/sos.h":

    cdef enum:
        SOS_VERS_MAJOR
        SOS_VERS_MINOR
        SOS_VERS_FIX

    cdef struct sos_attr_s
    cdef struct sos_index_s
    cdef struct sos_schema_s
    cdef struct sos_obj_s
    ctypedef sos_attr_s *sos_attr_t
    ctypedef sos_index_s *sos_index_t
    ctypedef sos_schema_s *sos_schema_t
    ctypedef sos_obj_s *sos_obj_t

    cdef enum:
        SOS_SCHEMA_NAME_LEN
        SOS_ATTR_NAME_LEN
        SOS_INDEX_NAME_LEN
        SOS_INDEX_KEY_TYPE_LEN
        SOS_INDEX_TYPE_LEN
        SOS_INDEX_ARGS_LEN

    cdef enum sos_type_e:
        SOS_TYPE_INT16,
        SOS_TYPE_FIRST,
        SOS_TYPE_INT32,
        SOS_TYPE_INT64,
        SOS_TYPE_UINT16,
        SOS_TYPE_UINT32,
        SOS_TYPE_UINT64,
        SOS_TYPE_FLOAT,
        SOS_TYPE_DOUBLE,
        SOS_TYPE_LONG_DOUBLE,
        SOS_TYPE_TIMESTAMP,
        SOS_TYPE_OBJ,
        SOS_TYPE_STRUCT,
        SOS_TYPE_BYTE_ARRAY,
        SOS_TYPE_ARRAY,
        SOS_TYPE_CHAR_ARRAY,
        SOS_TYPE_STRING,
        SOS_TYPE_INT16_ARRAY,
        SOS_TYPE_INT32_ARRAY,
        SOS_TYPE_INT64_ARRAY,
        SOS_TYPE_UINT16_ARRAY,
        SOS_TYPE_UINT32_ARRAY,
        SOS_TYPE_UINT64_ARRAY,
        SOS_TYPE_FLOAT_ARRAY,
        SOS_TYPE_DOUBLE_ARRAY,
        SOS_TYPE_LONG_DOUBLE_ARRAY,
        SOS_TYPE_OBJ_ARRAY,
        SOS_TYPE_JOIN,
        SOS_TYPE_LAST,
    ctypedef sos_type_e sos_type_t

    ctypedef ods_idx_data_t sos_idx_data_t

    cdef struct sos_idx_ref_s:
        uuid_t part_uuid
        ods_ref_t obj

    cdef union sos_obj_ref_s:
        ods_idx_data_t idx_data
        sos_idx_ref_s ref

    ctypedef sos_obj_ref_s sos_obj_ref_t

    void sos_ref_reset(sos_obj_ref_t ref)

    cdef union sos_array_element_u:
        char char_[0]
        uint8_t byte_[0]
        uint16_t uint16_[0]
        uint32_t uint32_[0]
        uint64_t uint64_[0]
        int16_t int16_[0]
        int32_t int32_[0]
        int64_t int64_[0]
        float float_[0]
        double double_[0]
        long double long_double_[0]
        sos_obj_ref_t ref_[0]
    ctypedef sos_array_element_u sos_array_element_t

    cdef struct sos_array_s:
        uint32_t count
        sos_array_element_t data
    ctypedef sos_array_s *sos_array_t

    cdef struct ods_timeval_s:
        uint32_t tv_usec
        uint32_t tv_sec

    cdef struct sos_timeval_s:
        uint32_t usecs
        uint32_t secs

    cdef union sos_timestamp_u:
        uint64_t time
        ods_timeval_s tv
        sos_timeval_s fine

    cdef union sos_primary_u:
        unsigned char byte_
        uint16_t uint16_
        uint32_t uint32_
        uint64_t uint64_
        int16_t int16_
        int32_t int32_
        int64_t int64_
        float float_
        double double_
        long double long_double_
        sos_timestamp_u timestamp_
        sos_obj_ref_s ref_
        uint8_t struc_[0]

    cdef union sos_value_data_u:
        sos_primary_u prim
        sos_array_element_u struc
        sos_array_s array
    ctypedef sos_value_data_u *sos_value_data_t

    cdef struct sos_value_s:
        sos_obj_t obj
        sos_attr_t attr
        sos_value_data_u data_
        sos_value_data_t data
    ctypedef sos_value_s *sos_value_t

    cdef enum sos_cond_e:
        SOS_COND_LT,
        SOS_COND_LE,
        SOS_COND_EQ,
        SOS_COND_GE,
        SOS_COND_GT,
        SOS_COND_NE,
    ctypedef sos_cond_e sos_cond_t

    cdef struct sos_schema_template_attr:
        const char *name
        sos_type_t type
        size_t size
        int indexed
        const char *idx_type
        const char *key_type
        const char *idx_args
    ctypedef sos_schema_template_attr *sos_schema_template_attr_t

    cdef struct sos_schema_template:
        const char *name
        sos_schema_template_attr attrs[0]
    ctypedef sos_schema_template *sos_schema_template_t

    cdef struct sos_container_s:
        pass
    ctypedef sos_container_s *sos_t

    cdef enum sos_perm_e:
        SOS_PERM_RD,
        SOS_PERM_WR,
        SOS_PERM_RW,
        SOS_PERM_CREAT,
        SOS_BE_MMOS,
        SOS_BE_LSOS,
        SOS_PERM_USER

    ctypedef sos_perm_e sos_perm_t

    cdef enum sos_commit_e:
        SOS_COMMIT_ASYNC,
        SOS_COMMIT_SYNC
    ctypedef sos_commit_e sos_commit_t

    cdef struct sos_config_iter_s:
        pass
    ctypedef sos_config_iter_s *sos_config_iter_t

    cdef enum:
        SOS_CONTAINER_NAME_LEN,
        SOS_CONFIG_NAME_LEN
    cdef struct sos_config_data_s:
        char name[SOS_CONFIG_NAME_LEN]
        char value[0]
    ctypedef sos_config_data_s *sos_config_t

    cdef enum:
        SOS_VER
    cdef struct sos_version_s:
        uint8_t major
        uint8_t minor
        uint16_t fix
        char *git_commit_id

    sos_version_s sos_container_version(sos_t c)
    int sos_container_new(const char *path, int o_mode)
    sos_t sos_container_open(const char *path, sos_perm_t o_perm, int o_mode)
    int sos_container_delete(sos_t c)
    int sos_container_stat(sos_t sos, stat *sb)
    void sos_container_close(sos_t c, sos_commit_t flags)
    int sos_container_commit(sos_t c, sos_commit_t flags)
    void sos_begin_x_wait(sos_t c, timespec *ts)
    void sos_begin_x(sos_t c)
    void sos_end_x(sos_t c)
    void sos_container_info(sos_t sos, FILE* fp)
    void sos_inuse_obj_info(sos_t sos, FILE *fp)
    void sos_free_obj_info(sos_t sos, FILE *fp)
    int sos_container_config_set(const char *path, const char *option, const char *value)
    char *sos_container_config_get(const char *path, const char *option)
    sos_config_iter_t sos_config_iter_new(const char *path)
    void sos_config_iter_free(sos_config_iter_t iter)
    sos_config_t sos_config_first(sos_config_iter_t iter)
    sos_config_t sos_config_next(sos_config_iter_t iter)
    void sos_config_print(const char *path, FILE *fp)

    int sos_schema_export(const char *path, FILE *)
    sos_schema_t sos_schema_create(const char *name, uuid_t uuid)
    sos_schema_t sos_schema_new(const char *name)
    void sos_schema_free(sos_schema_t schema)
    sos_schema_t sos_schema_dup(sos_schema_t schema)
    size_t sos_schema_count(sos_t sos)
    int sos_schema_add(sos_t sos, sos_schema_t schema)
    sos_schema_t sos_schema_from_template(sos_schema_template_t pt)
    sos_schema_t sos_schema_by_name(sos_t sos, const char *name)
    sos_schema_t sos_schema_by_uuid(sos_t sos, uuid_t uuid)
    void sos_schema_print(sos_schema_t schema, FILE *fp)
    int sos_schema_delete(sos_t sos, const char *name)
    sos_schema_t sos_schema_first(sos_t sos)
    sos_schema_t sos_schema_next(sos_schema_t schema)
    const char *sos_schema_name(sos_schema_t schema)
    void sos_schema_uuid(sos_schema_t schema, uuid_t uuid)
    uint32_t sos_schema_gen(sos_schema_t schema)
    int sos_schema_attr_count(sos_schema_t schema)
    int sos_schema_attr_add(sos_schema_t schema, const char *name, sos_type_t type, ...)
    int sos_schema_index_add(sos_schema_t schema, const char *name)
    int sos_schema_index_rem(sos_schema_t schema, const char *attr_name)
    int sos_schema_index_modify(sos_schema_t schema, const char *name,
                                const char *idx_type, const char *key_type, ...)
    sos_attr_t sos_schema_attr_by_name(sos_schema_t schema, const char *name)
    sos_attr_t sos_schema_attr_by_id(sos_schema_t schema, int attr_id)
    sos_attr_t sos_schema_attr_first(sos_schema_t schema)
    sos_attr_t sos_schema_attr_last(sos_schema_t schema)
    sos_attr_t sos_schema_attr_next(sos_attr_t attr)
    sos_attr_t sos_schema_attr_prev(sos_attr_t attr)

    int sos_attr_id(sos_attr_t attr)
    const char *sos_attr_name(sos_attr_t attr)
    sos_type_t sos_attr_type(sos_attr_t attr)
    sos_index_t sos_attr_index(sos_attr_t attr)
    const char *sos_attr_idx_type(sos_attr_t attr)
    const char *sos_attr_key_type(sos_attr_t attr)
    const char *sos_attr_idx_args(sos_attr_t attr)
    size_t sos_attr_size(sos_attr_t attr)
    sos_schema_t sos_attr_schema(sos_attr_t attr)
    int sos_attr_join(sos_obj_t obj, sos_attr_t attr)
    sos_array_t sos_attr_join_list(sos_attr_t attr)

    int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
                                      const char *attr_name, const char *attr_value,
                                      char **endptr)

    cdef enum:
        SOS_PART_NAME_DEFAULT,
        SOS_PART_NAME_LEN,
        SOS_PART_PATH_LEN,

    cdef enum sos_part_state_e:
        SOS_PART_STATE_OFFLINE,
        SOS_PART_STATE_ACTIVE,
        SOS_PART_STATE_PRIMARY,
        SOS_PART_STATE_BUSY,
        SOS_PART_STATE_DETACHED

    ctypedef sos_part_state_e sos_part_state_t

    cdef struct sos_part_stat_s:
        uint64_t size
        uint64_t accessed
        uint64_t modified
        uint64_t changed
    ctypedef sos_part_stat_s *sos_part_stat_t

    cdef struct sos_part_iter_s:
        pass
    ctypedef sos_part_iter_s *sos_part_iter_t
    cdef struct sos_part_s:
        pass
    ctypedef sos_part_s *sos_part_t

    cdef struct sos_part_uuid_entry_s:
        uuid_t uuid
        long count
    ctypedef sos_part_uuid_entry_s *sos_part_uuid_entry_t

    sos_part_t sos_part_open(const char *path, int o_perm, int o_mode, const char *desc)
    int sos_part_chown(sos_part_t part, uid_t uid, gid_t gid)
    int sos_part_chmod(sos_part_t part, int perm)
    int sos_part_attach(sos_t sos, const char *name, const char *path)
    int sos_part_detach(sos_t sos, const char *name)
    sos_part_t sos_part_by_name(sos_t sos, const char *name)
    sos_part_t sos_part_by_path(sos_t sos, const char *path)
    uid_t sos_part_uid(sos_part_t part)
    gid_t sos_part_gid(sos_part_t part)
    int sos_part_perm(sos_part_t part)

    sos_part_iter_t sos_part_iter_new(sos_t sos)
    void sos_part_iter_free(sos_part_iter_t iter)
    sos_part_t sos_part_first(sos_part_iter_t iter)
    sos_part_t sos_part_next(sos_part_iter_t iter)
    const char *sos_part_name(sos_part_t part)
    const char *sos_part_path(sos_part_t part)
    const char *sos_part_desc(sos_part_t part)
    void sos_part_desc_set(sos_part_t part, const char *name)
    void sos_part_uuid(sos_part_t part, uuid_t uuid)
    sos_part_state_t sos_part_state(sos_part_t part)
    int sos_part_state_set(sos_part_t part, sos_part_state_t state)
    uint32_t sos_part_refcount(sos_part_t part)
    void sos_part_put(sos_part_t part)
    int sos_part_stat(sos_part_t part, sos_part_stat_t stat)
    int sos_part_verify(char *path, FILE *fp)
    ctypedef int (*sos_part_obj_iter_fn_t)(sos_part_t part, sos_obj_t obj, void *arg)
    cdef struct sos_part_obj_iter_pos_s:
        pass
    ctypedef sos_part_obj_iter_pos_s *sos_part_obj_iter_pos_t

    void sos_part_obj_iter_pos_init(sos_part_obj_iter_pos_t pos)
    int sos_part_obj_iter(sos_part_t part, sos_part_obj_iter_pos_t pos,
                          sos_part_obj_iter_fn_t fn, void *arg)
    size_t sos_part_remap_schema_uuid(sos_part_t part, const char *dst_path, const char *src_path)
    sos_part_uuid_entry_t sos_part_query_schema_uuid(sos_part_t part, size_t *count)
    ctypedef int (*sos_part_reindex_callback_fn)(sos_part_t part, void *arg, uint64_t obj_count)
    size_t sos_part_reindex(sos_part_t part,
                            sos_part_reindex_callback_fn callback_fn, void *callback_arg,
                            size_t obj_count)

    cdef enum sos_byte_order_e:
        SOS_OBJ_BE,
        SOS_OBJ_LE

    sos_obj_t sos_obj_new(sos_schema_t schema)
    sos_obj_t sos_obj_new_with_data(sos_schema_t schema, uint8_t *data, size_t data_size)
    sos_obj_t sos_obj_malloc(sos_schema_t schema)
    sos_obj_t sos_obj_malloc_size(sos_schema_t schema, size_t reserve)
    int sos_obj_commit(sos_obj_t obj)
    sos_schema_t sos_obj_schema(sos_obj_t obj)

    sos_obj_ref_t sos_obj_ref(sos_obj_t obj)
    sos_obj_t sos_ref_as_obj(sos_t sos, sos_obj_ref_t ref)

    sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val)
    void sos_obj_delete(sos_obj_t obj)
    sos_obj_t sos_obj_get(sos_obj_t obj)
    void sos_obj_put(sos_obj_t obj)
    int sos_obj_index(sos_obj_t obj)
    int sos_obj_remove(sos_obj_t obj)
    sos_value_t sos_value_by_name(sos_value_t value, sos_schema_t schema, sos_obj_t obj,
                                  const char *name, int *attr_id)
    sos_value_t sos_value_by_id(sos_value_t value, sos_obj_t obj, int attr_id)
    sos_obj_t sos_array_obj_new(sos_t sos, sos_type_t type, size_t count)
    int sos_attr_is_ref(sos_attr_t attr)
    int sos_attr_is_array(sos_attr_t attr)
    size_t sos_array_count(sos_value_t val)
    sos_value_t sos_array_new(sos_value_t val, sos_attr_t attr, sos_obj_t obj, size_t count)
    sos_array_t sos_array(sos_value_t val)
    sos_value_t sos_value_new()
    void sos_value_free(sos_value_t v)
    void *sos_obj_ptr(sos_obj_t obj)
    sos_value_t sos_value_init(sos_value_t value, sos_obj_t obj, sos_attr_t attr)
    sos_value_t sos_value(sos_obj_t obj, sos_attr_t attr)
    sos_value_data_t sos_obj_attr_data(sos_obj_t obj, sos_attr_t attr, sos_obj_t *arr_obj)
    void sos_value_put(sos_value_t value)
    int sos_value_cmp(sos_value_t a, sos_value_t b)
    sos_type_t sos_value_type(sos_value_t value)
    const char *sos_value_type_name(sos_type_t t)
    size_t sos_value_size(sos_value_t value)
    sos_value_data_t sos_value_data_new(sos_type_t typ, size_t count)
    void sos_value_data_del(sos_value_data_t vd)
    size_t sos_value_memcpy(sos_value_t value, void *buf, size_t buflen)
    size_t sos_obj_attr_strlen(sos_obj_t obj, sos_attr_t attr)
    char *sos_obj_attr_to_str(sos_obj_t obj, sos_attr_t attr, char *str, size_t len)
    int sos_obj_attr_from_str(sos_obj_t obj, sos_attr_t attr, const char *str, char **endptr)
    size_t sos_value_strlen(sos_value_t v)
    const char *sos_value_to_str(sos_value_t value, char *str, size_t len)
    int sos_value_from_str(sos_value_t value, const char *str, char **endptr)
    int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
                                      const char *attr_name, const char *attr_value,
                                      char **endptr)
    char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj, const char *attr_name,
                                      char *str, size_t len)
    char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj, const char *attr_name,
				char *str, size_t len)
    char *sos_obj_attr_by_id_to_str(sos_obj_t sos_obj, int,
                                    char *str, size_t len);


    ctypedef ods_obj_s *sos_key_t

    cdef struct sos_comp_key_spec:
        int type
        sos_value_data_t data

    ctypedef sos_comp_key_spec *sos_comp_key_spec_t

    cdef struct sos_index_stat_s:
        uint64_t cardinality
        uint64_t duplicates
        uint64_t size
    ctypedef sos_index_stat_s *sos_index_stat_t

    ctypedef int (*sos_ins_cb_fn_t)(sos_index_t index, sos_key_t key,
                                    int missing, sos_obj_ref_t *ref, void *arg)

    cdef enum sos_visit_action:
        SOS_VISIT_ADD,
        SOS_VISIT_DEL,
        SOS_VISIT_UPD,
        SOS_VISIT_NOP
    ctypedef sos_visit_action sos_visit_action_t

    ctypedef sos_visit_action_t (*sos_visit_cb_fn_t)(sos_index_t index,
                                                     sos_key_t key, sos_idx_data_t *idx_data,
                                                     int found,
                                                     void *arg)
    sos_obj_t sos_obj_find(sos_attr_t attr, sos_key_t key)
    int sos_index_new(sos_t sos, const char *name,
                      const char *idx_type, const char *key_type,
                      const char *args)
    sos_index_t sos_index_open(sos_t sos, const char *name)
    int sos_index_insert(sos_index_t index, sos_key_t key, sos_obj_t obj)
    int sos_index_remove(sos_index_t index, sos_key_t key, sos_obj_t obj)
    int sos_index_visit(sos_index_t index, sos_key_t key, sos_visit_cb_fn_t cb_fn, void *arg)
    sos_obj_t sos_index_find(sos_index_t index, sos_key_t key)
    sos_obj_t sos_index_find_inf(sos_index_t index, sos_key_t key)
    sos_obj_t sos_index_find_sup(sos_index_t index, sos_key_t key)
    sos_obj_t sos_index_find_le(sos_index_t index, sos_key_t key)
    sos_obj_t sos_index_find_ge(sos_index_t index, sos_key_t key)
    sos_obj_t sos_index_find_max(sos_index_t index, sos_key_t *pkey)
    sos_obj_t sos_index_find_min(sos_index_t index, sos_key_t *pkey)
    int sos_index_commit(sos_index_t index, sos_commit_t flags)
    int sos_index_close(sos_index_t index, sos_commit_t flags)
    size_t sos_index_key_size(sos_index_t index)
    sos_key_t sos_index_key_new(sos_index_t index, size_t size)
    int sos_index_key_from_str(sos_index_t index, sos_key_t key, const char *str)
    const char *sos_index_key_to_str(sos_index_t index, sos_key_t key)
    int sos_index_key_cmp(sos_index_t index, sos_key_t a, sos_key_t b)
    void sos_index_print(sos_index_t index, FILE *fp)
    int sos_index_verify(sos_index_t index, FILE *fp, int verify)
    const char *sos_index_name(sos_index_t index)
    int sos_index_stat(sos_index_t index, sos_index_stat_t sb)
    void sos_container_index_list(sos_t sos, FILE *fp)

    cdef struct sos_container_index_iter_s:
        pass
    ctypedef sos_container_index_iter_s *sos_container_index_iter_t

    sos_container_index_iter_t sos_container_index_iter_new(sos_t sos)
    void sos_container_index_iter_free(sos_container_index_iter_t iter)
    sos_index_t sos_container_index_iter_first(sos_container_index_iter_t iter)
    sos_index_t sos_container_index_iter_next(sos_container_index_iter_t iter)

    sos_key_t sos_key_new(size_t sz)
    void sos_key_put(sos_key_t key)
    size_t sos_key_set(sos_key_t key, void *value, size_t sz)
    char *sos_key_to_str(sos_key_t key, const char *fmt, const char *sep, size_t el_sz)
    size_t sos_key_size(sos_key_t key)
    size_t sos_key_len(sos_key_t key)
    unsigned char *sos_key_value(sos_key_t key)
    void *sos_value_as_key(sos_value_t value)
    int sos_comp_key_set(sos_key_t key, size_t len, sos_comp_key_spec_t key_spec)
    sos_comp_key_spec_t sos_comp_key_get(sos_key_t key, size_t *len)
    size_t sos_comp_key_size(size_t len, sos_comp_key_spec_t key_spec)

    int sos_attr_key_from_str(sos_attr_t attr, sos_key_t key, const char *str)
    const char *sos_attr_key_to_str(sos_attr_t attr, sos_key_t key)
    sos_key_t sos_attr_key_new(sos_attr_t attr, size_t len)
    int sos_attr_key_cmp(sos_attr_t attr, sos_key_t a, sos_key_t b)
    size_t sos_attr_key_size(sos_attr_t attr)

    cdef struct sos_iter_s:
        pass
    ctypedef sos_iter_s *sos_iter_t
    ctypedef uint32_t sos_pos_t

    cdef enum sos_iter_flags_e:
        SOS_ITER_F_ALL,
        SOS_ITER_F_UNIQUE,
        SOS_ITER_F_INF_LAST_DUP,
        SOS_ITER_F_SUP_LAST_DUP,
        SOS_ITER_F_MASK,
    ctypedef uint64_t sos_iter_flags_t

    cdef struct sos_filter_cond_s:
        pass
    ctypedef sos_filter_cond_s *sos_filter_cond_t
    cdef struct sos_filter_s:
        pass
    ctypedef sos_filter_s *sos_filter_t

    sos_iter_t sos_index_iter_new(sos_index_t index)
    sos_iter_t sos_attr_iter_new(sos_attr_t attr)
    void sos_iter_free(sos_iter_t iter)
    int sos_iter_key_cmp(sos_iter_t iter, sos_key_t other)
    int sos_iter_find(sos_iter_t iter, sos_key_t key)
    int sos_iter_find_first(sos_iter_t iter, sos_key_t key)
    int sos_iter_find_last(sos_iter_t iter, sos_key_t key)
    int sos_iter_inf(sos_iter_t i, sos_key_t key)
    int sos_iter_sup(sos_iter_t i, sos_key_t key)
    int sos_iter_find_le(sos_iter_t i, sos_key_t key)
    int sos_iter_find_ge(sos_iter_t i, sos_key_t key)

    int sos_iter_flags_set(sos_iter_t i, sos_iter_flags_t flags)
    sos_iter_flags_t sos_iter_flags_get(sos_iter_t i)
    uint64_t sos_iter_card(sos_iter_t i)
    uint64_t sos_iter_dups(sos_iter_t i)
    int sos_iter_next(sos_iter_t iter)
    int sos_iter_prev(sos_iter_t i)
    int sos_iter_begin(sos_iter_t i)
    int sos_iter_end(sos_iter_t i)
    sos_key_t sos_iter_key(sos_iter_t iter)
    sos_obj_t sos_iter_obj(sos_iter_t iter)
    sos_obj_ref_t sos_iter_ref(sos_iter_t iter)
    int sos_iter_entry_remove(sos_iter_t iter)

    sos_filter_t sos_filter_new(sos_iter_t iter)
    void sos_filter_free(sos_filter_t f)
    int sos_filter_cond_add(sos_filter_t f, sos_attr_t attr,
                            sos_cond_t cond_e, sos_value_t value)
    sos_obj_t sos_filter_begin(sos_filter_t filt)
    sos_obj_t sos_filter_next(sos_filter_t filt)
    sos_obj_t sos_filter_prev(sos_filter_t filt)
    sos_obj_t sos_filter_end(sos_filter_t filt)
    int sos_filter_miss_count(sos_filter_t filt)
    sos_obj_t sos_filter_obj(sos_filter_t filt)
    int sos_filter_flags_set(sos_filter_t filt, sos_iter_flags_t flags)
    sos_iter_flags_t sos_filter_flags_get(sos_filter_t filt)

cdef extern from "dsos.h":

    cdef enum dsos_error:
        DSOS_ERR_OK,
        DSOS_ERR_MEMORY,
        DSOS_ERR_CLIENT,
        DSOS_ERR_SCHEMA,
        DSOS_ERR_ATTR,
        DSOS_ERR_ITER,
        DSOS_ERR_ITER_EMPTY,
        DSOS_ERR_QUERY_ID,
        DSOS_ERR_QUERY_EMPTY,
        DSOS_ERR_QUERY_BAD_SELECT,
        DSOS_ERR_PARAMETER,
        DSOS_ERR_TRANSPORT

    cdef struct dsos_result_s:
        int count
        dsos_error any_err
        dsos_error res[16]
    ctypedef dsos_result_s dsos_res_t

    cdef struct dsos_container_s:
        pass
    cdef struct dsos_session_s:
        pass
    cdef struct dsos_schema_s:
        pass
    cdef struct dsos_iter_s:
        pass
    cdef struct dsos_query_s:
        pass
    cdef struct dsos_session_s:
        pass
    cdef struct dsos_part_s:
        pass
    ctypedef dsos_session_s *dsos_session_t
    ctypedef dsos_container_s *dsos_container_t
    ctypedef dsos_schema_s *dsos_schema_t
    ctypedef dsos_part_s *dsos_part_t
    ctypedef dsos_iter_s *dsos_iter_t
    ctypedef dsos_query_s *dsos_query_t
    cdef struct dsos_name_array_s:
        int count
        char **names
    ctypedef dsos_name_array_s *dsos_name_array_t

    const char *dsos_last_errmsg()
    const int dsos_last_error()

    void dsos_session_close(dsos_session_t sess)
    dsos_session_t dsos_session_open(const char *config_file)
    void dsos_container_close(dsos_container_t cont)
    void dsos_container_commit(dsos_container_t cont)
    dsos_container_t dsos_container_open(dsos_session_t sess, const char *path, sos_perm_t perm, int mode)
    int dsos_container_error(dsos_container_t cont, const char **err_msg)
    dsos_schema_t dsos_schema_create(dsos_container_t cont, sos_schema_t schema, dsos_res_t *res)
    dsos_name_array_t dsos_schema_query(dsos_container_t cont)
    dsos_schema_t dsos_schema_by_name(dsos_container_t cont, const char *name)
    dsos_schema_t dsos_schema_by_uuid(dsos_container_t cont, uuid_t uuid)
    sos_schema_t dsos_schema_local(dsos_schema_t schema)
    void dsos_schema_print(dsos_schema_t schema, FILE *fp)

    dsos_part_t dsos_part_create(dsos_container_t cont,
    	const char *name, const char *desc, int mode,
	    uid_t uid, gid_t gid, int perm)
    dsos_name_array_t dsos_part_query(dsos_container_t cont)
    void dsos_name_array_free(dsos_name_array_t names)
    dsos_part_t dsos_part_by_name(dsos_container_t cont, const char *name)
    dsos_part_t dsos_part_by_uuid(dsos_container_t cont, const uuid_t uuid)
    const char *dsos_part_name(dsos_part_t part)
    const char *dsos_part_desc(dsos_part_t part)
    const char *dsos_part_path(dsos_part_t part)
    sos_part_state_t dsos_part_state(dsos_part_t part)
    void dsos_part_uuid(dsos_part_t part, uuid_t uuid)
    uid_t dsos_part_uid(dsos_part_t part)
    gid_t dsos_part_gid(dsos_part_t part)
    int dsos_part_perm(dsos_part_t part)
    int dsos_part_state_set(dsos_part_t part, int state)
    int dsos_part_chown(dsos_part_t part, uid_t uid, gid_t gid)
    int dsos_part_chmod(dsos_part_t part, int mode)

    int dsos_schema_attr_count(dsos_schema_t schema)
    sos_attr_t dsos_schema_attr_by_id(dsos_schema_t schema, int attr_id)
    sos_attr_t dsos_schema_attr_by_name(dsos_schema_t schema, const char *name)
    int dsos_transaction_begin(dsos_container_t cont, timespec *ts)
    int dsos_transaction_end(dsos_container_t cont)
    int dsos_obj_create(dsos_container_t cont, dsos_part_t part, dsos_schema_t schema, sos_obj_t obj)
    dsos_iter_t dsos_iter_create(dsos_container_t cont, dsos_schema_t schema, const char *attr_name)
    sos_obj_t dsos_iter_begin(dsos_iter_t iter)
    sos_obj_t dsos_iter_end(dsos_iter_t iter)
    sos_obj_t dsos_iter_next(dsos_iter_t iter)
    sos_obj_t dsos_iter_prev(dsos_iter_t iter)
    sos_obj_t dsos_iter_find_glb(dsos_iter_t iter, sos_key_t key)
    sos_obj_t dsos_iter_find_lub(dsos_iter_t iter, sos_key_t key)
    sos_obj_t dsos_iter_find_le(dsos_iter_t iter, sos_key_t key)
    sos_obj_t dsos_iter_find_ge(dsos_iter_t iter, sos_key_t key)
    sos_obj_t dsos_iter_find(dsos_iter_t iter, sos_key_t key)
    int dsos_attr_value_min(dsos_container_t cont, sos_attr_t attr)
    int dsos_attr_value_max(dsos_container_t cont, sos_attr_t attr)
    dsos_query_t dsos_query_create(dsos_container_t cont)
    void dsos_query_destroy(dsos_query_t query)
    int dsos_query_select(dsos_query_t query, const char *clause)
    const char *dsos_query_errmsg(dsos_query_t query)
    sos_obj_t dsos_query_next(dsos_query_t query)
    sos_schema_t dsos_query_schema(dsos_query_t query)
    void dsos_name_array_free(dsos_name_array_t name)

