typedef unsigned long dsos_container_id;
union dsos_open_res switch (int error) {
    case 0:
        dsos_container_id cont;
    default:
	string error_msg<>;
};

typedef unsigned long dsos_obj_id;
typedef unsigned char dsos_obj_value<>;

/* Result for API that return objects */
union dsos_create_res switch (int error) {
    case 0:
        dsos_obj_id obj_id;   /* object handle(s) unique for this session */
    default:
        void;
};

/* Result for API that return an iterator */
typedef unsigned long dsos_iter_id;
union dsos_iter_res switch (int error) {
    case 0:
        dsos_iter_id iter_id;
    default:
        void;
};

typedef unsigned char dsos_uuid[16];
typedef string dsos_uuid_str<48>;
typedef string dsos_schema_name<255>;
typedef string dsos_attr_name<255>;
typedef unsigned long dsos_schema_id;

/* Type for API that return an object list */
typedef struct dsos_obj_entry *dsos_obj_link;
struct dsos_obj_entry {
    dsos_container_id cont_id;
    dsos_schema_id schema_id;
    dsos_obj_value value;
    dsos_obj_link next;
};

union dsos_obj_list_res switch (int error) {
    case 0:
        dsos_obj_link obj_list;
    default:
        void;
};

struct dsos_attr_spec {
    dsos_attr_name name;
    int id;
};

struct dsos_schema_spec_attr {
	dsos_schema_name name;
	int type;
	int size;
	dsos_attr_spec join_list<>;
	int indexed;
	string idx_type<128>;
	string key_type<128>;
	string idx_args<128>;
};

struct dsos_schema_spec {
	dsos_schema_name name;  /* Unique for the container */
    dsos_schema_id id;      /* Session local id, returned on create/find */
    dsos_uuid uuid;         /* Universally unique id for the schema */
    dsos_schema_spec_attr attrs<>;
};

/* Result for API that return schema */
union dsos_schema_res switch (int error) {
    case 0:
        dsos_schema_spec *spec;
    default:
        void;
};

typedef int dsos_schema_attr;
union dsos_schema_attr_res switch (int error) {
    case 0:
        dsos_schema_attr attr;
    default:
        void;
};


typedef string dsos_name<>;

union dsos_schema_query_res switch (int error) {
    case 0:
        dsos_name names<>;
    default:
        void;
};

/* Result for API that return a query */
typedef string dsos_query<>;
typedef int dsos_query_result_size;
typedef unsigned long dsos_query_id;
typedef int dsos_query_result_format;
union dsos_query_create_res switch (int error) {
    case 0:
        dsos_query_id query_id;
    default:
        string error_msg<>;
};

typedef string dsos_query_error<>;
struct dsos_query_select_data {
    int key_attr_id;        /* The id of the attribute used to order the data */
    dsos_schema_spec *spec; /* schema specification */
};
union dsos_query_select_res switch (int error) {
    case 0:
    	dsos_query_select_data select;
    default:
        dsos_query_error error_msg;
};

struct dsos_query_next_result {
    int format; /* The format of the data in the object list */
    int count;  /* Number of records in the object list */
    dsos_obj_link obj_list;
};

union dsos_query_next_res switch (int error) {
    case 0:
        dsos_query_next_result result;
    default:
        dsos_query_error error_msg;
};

const QUERY_RESULT_LIMIT = 1;
const QUERY_RESULT_FORMAT = 2;
typedef int query_option;
typedef string query_option_type_string<>;
typedef int query_option_type_int;

union dsos_query_option_value switch (int query_option_type) {
    case QUERY_RESULT_FORMAT:
        query_option_type_string format;
    case QUERY_RESULT_LIMIT:
        query_option_type_int limit;
    default:
        void;
};
typedef dsos_query_option_value dsos_query_options<>;

union dsos_query_destroy_res switch (int error) {
    case 0:
    	void;
    default:
        dsos_query_error error_msg;
};

program SOSDB {
    version SOSVERS {
        /* Container operations */
        dsos_open_res OPEN(string, int, int) = 10;
        int CLOSE(dsos_container_id) = 11;
        int COMMIT(dsos_container_id) = 12;
        int TRANSACTION_BEGIN(dsos_container_id) = 13;
        int TRANSACTION_END(dsos_container_id) = 14;

        /* Schema operations */
        dsos_schema_res SCHEMA_CREATE(dsos_container_id, dsos_schema_spec) = 20;
        dsos_schema_res SCHEMA_FIND_BY_ID(dsos_container_id, dsos_schema_id) = 21;
        dsos_schema_res SCHEMA_FIND_BY_NAME(dsos_container_id, string) = 22;
        dsos_schema_res SCHEMA_FIND_BY_UUID(dsos_container_id, string) = 23;
        dsos_schema_query_res SCHEMA_QUERY(dsos_container_id) = 64;

        /* Object operations */
        dsos_create_res OBJ_CREATE(dsos_obj_link) = 30;
        int OBJ_DELETE(dsos_container_id, dsos_obj_id) = 31;

        /* Iterator operations */
        dsos_iter_res ITER_CREATE(dsos_container_id, dsos_schema_id, dsos_attr_name) = 40;
        int ITER_DELETE(dsos_container_id, dsos_iter_id) = 41;
        dsos_obj_list_res ITER_BEGIN(dsos_container_id, dsos_iter_id) = 42;
        dsos_obj_list_res ITER_END(dsos_container_id, dsos_iter_id) = 43;
        dsos_obj_list_res ITER_FIND_GLB(dsos_container_id, dsos_iter_id) = 44;
        dsos_obj_list_res ITER_FIND_LUB(dsos_container_id, dsos_iter_id) = 45;
        dsos_obj_list_res ITER_NEXT(dsos_container_id, dsos_iter_id) = 46;
        dsos_obj_list_res ITER_PREV(dsos_container_id, dsos_iter_id) = 47;
        dsos_obj_list_res ITER_FIND(dsos_container_id, dsos_iter_id) = 48;

        /* Query operations */
        dsos_query_create_res QUERY_CREATE(dsos_container_id, dsos_query_options) = 60;
        dsos_query_select_res QUERY_SELECT(dsos_container_id, dsos_query_id, dsos_query) = 61;
        dsos_query_next_res QUERY_NEXT(dsos_container_id, dsos_query_id) = 62;
        dsos_query_destroy_res QUERY_DESTROY(dsos_container_id, dsos_query_id) = 63;
    } = 1;
} = 40009862;
