/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
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
%module bwx
%include "cpointer.i"
%include "cstring.i"
%include "carrays.i"
%include "exception.i"
%{
#include <stdio.h>
#include <sys/queue.h>
#include <sos/sos.h>
#include <bwx/bwx.h>

job_sample_t job_sample(sos_obj_t obj)
{
	return (job_sample_t)sos_obj_ptr(obj);
}

struct name_map_s {
	const char *name;
	int id;
} name_map[] = {
	{ "CompId", 					2 },
	{ "JobTime", 					1 },
	{ "RDMA_nrx", 18 },
	{ "RDMA_ntx", 20 },
	{ "RDMA_rx_bytes", 17 },
	{ "RDMA_tx_bytes", 19 },
	{ "SAMPLE_bteout_optA (B/s)", 130 },
	{ "SAMPLE_bteout_optB (B/s)", 129 },
	{ "SAMPLE_fmaout (B/s)", 131 },
	{ "SAMPLE_totalinput (B/s)", 132 },
	{ "SAMPLE_totaloutput_optA (B/s)", 133 },
	{ "SAMPLE_totaloutput_optB (B/s)", 128 },
	{ "SMSG_nrx", 22 },
	{ "SMSG_ntx", 24 },
	{ "SMSG_rx_bytes", 21 },
	{ "SMSG_tx_bytes", 23 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_device_memory",	7 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_l1_cache",	9 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_l2_cache",	8 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_register_file",	6 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_texture_memory", 5 },
	{ "Tesla_K20X.gpu_agg_dbl_ecc_total_errors",	4 },
	{ "Tesla_K20X.gpu_memory_used",		10 },
	{ "Tesla_K20X.gpu_power_limit", 13 },
	{ "Tesla_K20X.gpu_power_usage", 14 },
	{ "Tesla_K20X.gpu_pstate", 12 },
	{ "Tesla_K20X.gpu_temp",			11 },
	{ "Tesla_K20X.gpu_util_rate",			3 },
	{ "Time", 					0 },
	{ "X+_SAMPLE_GEMINI_LINK_BW (B/s)", 169 },
	{ "X+_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 145 },
	{ "X+_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 151 },
	{ "X+_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 157 },
	{ "X+_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 163 },
	{ "X+_credit_stall (ns)", 187 },
	{ "X+_inq_stall (ns)", 193 },
	{ "X+_packets (1)", 199 },
	{ "X+_recvlinkstatus (1)", 175 },
	{ "X+_sendlinkstatus (1)", 181 },
	{ "X+_traffic (B)", 205 },
	{ "X-_SAMPLE_GEMINI_LINK_BW (B/s)", 168 },
	{ "X-_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 144 },
	{ "X-_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 150 },
	{ "X-_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 156 },
	{ "X-_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 162 },
	{ "X-_credit_stall (ns)", 186 },
	{ "X-_inq_stall (ns)", 192 },
	{ "X-_packets (1)", 198 },
	{ "X-_recvlinkstatus (1)", 174 },
	{ "X-_sendlinkstatus (1)", 180 },
	{ "X-_traffic (B)", 204 },
	{ "Y+_SAMPLE_GEMINI_LINK_BW (B/s)", 167 },
	{ "Y+_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 143 },
	{ "Y+_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 149 },
	{ "Y+_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 155 },
	{ "Y+_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 161 },
	{ "Y+_credit_stall (ns)", 185 },
	{ "Y+_inq_stall (ns)", 191 },
	{ "Y+_packets (1)", 197 },
	{ "Y+_recvlinkstatus (1)", 173 },
	{ "Y+_sendlinkstatus (1)", 179 },
	{ "Y+_traffic (B)", 203 },
	{ "Y-_SAMPLE_GEMINI_LINK_BW (B/s)", 166 },
	{ "Y-_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 142 },
	{ "Y-_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 148 },
	{ "Y-_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 154 },
	{ "Y-_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 160 },
	{ "Y-_credit_stall (ns)", 184 },
	{ "Y-_inq_stall (ns)", 190 },
	{ "Y-_packets (1)", 196 },
	{ "Y-_recvlinkstatus (1)", 172 },
	{ "Y-_sendlinkstatus (1)", 178 },
	{ "Y-_traffic (B)", 202 },
	{ "Z+_SAMPLE_GEMINI_LINK_BW (B/s)", 165 },
	{ "Z+_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 141 },
	{ "Z+_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 147 },
	{ "Z+_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 153 },
	{ "Z+_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 159 },
	{ "Z+_credit_stall (ns)", 183 },
	{ "Z+_inq_stall (ns)", 189 },
	{ "Z+_packets (1)", 195 },
	{ "Z+_recvlinkstatus (1)", 171 },
	{ "Z+_sendlinkstatus (1)", 177 },
	{ "Z+_traffic (B)", 201 },
	{ "Z-_SAMPLE_GEMINI_LINK_BW (B/s)", 164 },
	{ "Z-_SAMPLE_GEMINI_LINK_CREDIT_STALL (% x1e6)", 140 },
	{ "Z-_SAMPLE_GEMINI_LINK_INQ_STALL (% x1e6)", 146 },
	{ "Z-_SAMPLE_GEMINI_LINK_PACKETSIZE_AVE (B)", 152 },
	{ "Z-_SAMPLE_GEMINI_LINK_USED_BW (% x1e6)", 158 },
	{ "Z-_credit_stall (ns)", 182 },
	{ "Z-_inq_stall (ns)", 188 },
	{ "Z-_packets (1)", 194 },
	{ "Z-_recvlinkstatus (1)", 170 },
	{ "Z-_sendlinkstatus (1)", 176 },
	{ "Z-_traffic (B)", 200 },
	{ "alloc_inode#stats.snx11001", 41 },
	{ "alloc_inode#stats.snx11002", 73 },
	{ "alloc_inode#stats.snx11003", 105 },
	{ "brw_read#stats.snx11001", 55},
	{ "brw_read#stats.snx11002", 87 },
	{ "brw_read#stats.snx11003", 119 },
	{ "brw_write#stats.snx11001", 54 },
	{ "brw_write#stats.snx11002", 86 },
	{ "brw_write#stats.snx11003", 118 },
	{ "bteout_optA", 136 },
	{ "bteout_optB", 135 },
	{ "close#stats.snx11001", 51 },
	{ "close#stats.snx11002", 83 },
	{ "close#stats.snx11003", 115 },
	{ "current_freemem", 25 },
	{ "direct_read#stats.snx11001", 35 },
	{ "direct_read#stats.snx11002", 67 },
	{ "direct_read#stats.snx11003", 99 },
	{ "direct_write#stats.snx11001", 34 },
	{ "direct_write#stats.snx11002", 66 },
	{ "direct_write#stats.snx11003", 98 },
	{ "dirty_pages_hits#stats.snx11001", 63 },
	{ "dirty_pages_hits#stats.snx11002", 95 },
	{ "dirty_pages_hits#stats.snx11003", 127 },
	{ "dirty_pages_misses#stats.snx11001", 62 },
	{ "dirty_pages_misses#stats.snx11002", 94 },
	{ "dirty_pages_misses#stats.snx11003", 126 },
	{ "flock#stats.snx11001", 44 },
	{ "flock#stats.snx11002", 76 },
	{ "flock#stats.snx11003", 108 },
	{ "fmaout", 137 },
	{ "fsync#stats.snx11001", 48 },
	{ "fsync#stats.snx11002", 80 },
	{ "fsync#stats.snx11003", 112 },
	{ "getattr#stats.snx11001", 43 },
	{ "getattr#stats.snx11002", 75 },
	{ "getattr#stats.snx11003", 107 },
	{ "getxattr#stats.snx11001", 39 },
	{ "getxattr#stats.snx11002", 71 },
	{ "getxattr#stats.snx11003", 103 },
	{ "inode_permission#stats.snx11001", 36 },
	{ "inode_permission#stats.snx11002", 68 },
	{ "inode_permission#stats.snx11003", 100 },
	{ "ioctl#stats.snx11001", 53 },
	{ "ioctl#stats.snx11002", 85 },
	{ "ioctl#stats.snx11003", 117 },
	{ "ipogif0_rx_bytes", 16 },
	{ "ipogif0_tx_bytes", 15 },
	{ "listxattr#stats.snx11001", 38 },
	{ "listxattr#stats.snx11002", 70 },
	{ "listxattr#stats.snx11003", 102 },
	{ "loadavg_5min(x100)", 28 },
	{ "loadavg_latest(x100)", 29 },
	{ "loadavg_running_processes", 27 },
	{ "loadavg_total_processes", 26 },
	{ "lockless_read_bytes#stats.snx11001", 33 },
	{ "lockless_read_bytes#stats.snx11002", 65 },
	{ "lockless_read_bytes#stats.snx11003", 97 },
	{ "lockless_truncate#stats.snx11001", 45 },
	{ "lockless_truncate#stats.snx11002", 77 },
	{ "lockless_truncate#stats.snx11003", 109 },
	{ "lockless_write_bytes#stats.snx11001", 32 },
	{ "lockless_write_bytes#stats.snx11002", 64 },
	{ "lockless_write_bytes#stats.snx11003", 96 },
	{ "mmap#stats.snx11001", 50 },
	{ "mmap#stats.snx11002", 82 },
	{ "mmap#stats.snx11003", 114},
	{ "nettopo_mesh_coord_X", 208 },
	{ "nettopo_mesh_coord_Y", 207 },
	{ "nettopo_mesh_coord_Z", 206 },
	{ "nr_dirty", 31 },
	{ "nr_writeback", 30 },
	{ "open#stats.snx11001", 52 },
	{ "open#stats.snx11002", 84 },
	{ "open#stats.snx11003", 116 },
	{ "read_bytes#stats.snx11001", 57 },
	{ "read_bytes#stats.snx11002", 89 },
	{ "read_bytes#stats.snx11003", 121 },
	{ "removexattr#stats.snx11001", 37 },
	{ "removexattr#stats.snx11002", 69},
	{ "removexattr#stats.snx11003", 101 },
	{ "seek#stats.snx11001", 49 },
	{ "seek#stats.snx11002", 81 },
	{ "seek#stats.snx11003", 113},
	{ "setattr#stats.snx11001", 47},
	{ "setattr#stats.snx11002", 79 },
	{ "setattr#stats.snx11003", 111 },
	{ "setxattr#stats.snx11001", 40 },
	{ "setxattr#stats.snx11002", 72 },
	{ "setxattr#stats.snx11003", 104 },
	{ "statfs#stats.snx11001", 42 },
	{ "statfs#stats.snx11002", 74 },
	{ "statfs#stats.snx11003", 106 },
	{ "totalinput", 138 },
	{ "totaloutput_optA", 139 },
	{ "totaloutput_optB", 134 },
	{ "truncate#stats.snx11001", 46 },
	{ "truncate#stats.snx11002", 78 },
	{ "truncate#stats.snx11003", 110 },
	{ "write_bytes#stats.snx11001", 56 },
	{ "write_bytes#stats.snx11002", 88 },
	{ "write_bytes#stats.snx11003", 120 },
	{ "writeback_failed_pages#stats.snx11001", 58 },
	{ "writeback_failed_pages#stats.snx11002", 90 },
	{ "writeback_failed_pages#stats.snx11003", 122 },
	{ "writeback_from_pressure#stats.snx11001", 60 },
	{ "writeback_from_pressure#stats.snx11002", 92 },
	{ "writeback_from_pressure#stats.snx11003", 124 },
	{ "writeback_from_writepage#stats.snx11001", 61 },
	{ "writeback_from_writepage#stats.snx11002", 93},
	{ "writeback_from_writepage#stats.snx11003", 125 },
	{ "writeback_ok_pages#stats.snx11001", 59 },
	{ "writeback_ok_pages#stats.snx11002", 91 },
	{ "writeback_ok_pages#stats.snx11003", 123 },
};

int cmp_names(const void *_a, const void *_b)
{
	const struct name_map_s *a = _a;
	const struct name_map_s *b = _b;
	return strcmp(a->name, b->name);
}

static int map(const char *name)
{
	struct name_map_s a = { name, 0 };
	struct name_map_s *res = bsearch(&a, name_map,
					 sizeof(name_map) / sizeof(name_map[0]),
					 sizeof(name_map[0]),
					 cmp_names);
	if (res)
		return res->id;
	return -1;
}

static uint64_t getval(struct job_sample_s *sample, size_t i)
{
	if (i < 0)
		return -1L;
	if (i >= sizeof(name_map) / sizeof(name_map[0]))
		return -1L;
	switch (i) {
	case 0:
		return *(uint64_t *)&sample->Time;
	case 2:
		return *(uint64_t *)&sample->JobTime;
	case 1:
		return (uint64_t)*(uint32_t *)&sample->CompId;
	default:
		return ((uint64_t *)&sample->Tesla_K20X_gpu_util_rate)[i-3];
	}
}

%}

typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef struct sos_timestamp_s {
	uint32_t usecs;
	uint32_t secs;
} *sos_timestamp_t;
typedef struct sos_container_s *sos_t;
typedef struct sos_obj_s *sos_obj_t;

%include <bwx/bwx.h>

job_sample_t job_sample(sos_obj_t obj);

%extend job_metric_vector_s {
	inline size_t __len__() { return self->count; }
	inline struct job_metric_s __getitem__(size_t i) {
		return self->vec[i];
	}
	inline const char *name(size_t i) {
		return sos_attr_name(self->vec[i].attr);
	}
 }

%extend job_sample_s {
	inline void __del__() {
		printf("Doh!!!\n");
	}
	inline size_t __len__() { return sizeof(name_map) / sizeof(name_map[0]); }
	inline uint64_t __getattr__(const char *name) {
		size_t i = map(name);
		if (i < 0)
			SWIG_exception(SWIG_IndexError, name);
		return getval(self, i);
	fail:
		return -1L;
	}
	inline uint64_t __getitem__(size_t i) {
		if (i < 0 || i >= sizeof(name_map) / sizeof(name_map[0]))
			SWIG_exception(SWIG_IndexError, "Index out of range");
		return getval(self, i);
	fail:
		return -1L;
	}
	inline int idx(const char *name) {
		return map(name);
	}
}

%extend job_time_key_s {
	inline uint64_t __getattr__(const char *name) {
		SWIG_exception(SWIG_IndexError, name);
	fail:
		return -1L;
	}
	inline uint64_t __getitem__(size_t i) {
		SWIG_exception(SWIG_IndexError, "Index out of bounds");
	fail:
		return -1L;
	}
}
%pythoncode %{
%}
