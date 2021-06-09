typedef unsigned long dsos_container_id;
union dsos_open_res switch (int error) {
    case 0:
        dsos_container_id cont;
    default:
        void;
};

typedef unsigned long dsos_obj_id;
typedef unsigned char dsos_obj_value<>;

/* Result for API that return objects */
union dsos_create_res switch (int error) {
    case 0:
        dsos_obj_id obj;   /* object handle(s) unique for this session */
    default:
        void;
};

/* Result for API that return an iterator */
typedef unsigned long dsos_iter_id;
union dsos_iter_res switch (int error) {
    case 0:
        dsos_iter_id iter;
    default:
        void;
};

typedef unsigned char dsos_uuid[16];
typedef string dsos_uuid_str<48>;
typedef string dsos_schema_name<255>;
typedef string dsos_attr_name<255>;
typedef unsigned long dsos_schema_id;

/* Return type for API that return an object list */
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

program SOSDB {
    version SOSVERS {
        /* Container operations */
        dsos_open_res OPEN(string, int, int) = 10;
        int CLOSE(dsos_container_id) = 11;
        int COMMIT(dsos_container_id) = 12;
        int TRANSACTION_BEGIN(dsos_container_id) = 13;
        int TRANSACTION_END(dsos_container_id) = 14;

        /* Schema operations */
        dsos_schema_res SCHEMA_CREATE(dsos_container_id, dsos_schema_spec) = 40;
        dsos_schema_res SCHEMA_FIND_BY_ID(dsos_container_id, dsos_schema_id) = 41;
        dsos_schema_res SCHEMA_FIND_BY_NAME(dsos_container_id, string) = 42;
        dsos_schema_res SCHEMA_FIND_BY_UUID(dsos_container_id, string) = 43;

        /* Object operations */
        dsos_create_res OBJ_CREATE(dsos_obj_link) = 20;
        int OBJ_DELETE(dsos_container_id, dsos_obj_id) = 21;

        /* Iterator operations */
        dsos_iter_res ITER_CREATE(dsos_container_id, dsos_schema_id, dsos_attr_name) = 30;
        int ITER_DELETE(dsos_container_id, dsos_iter_id) = 31;
        dsos_obj_list_res ITER_BEGIN(dsos_container_id, dsos_iter_id) = 32;
        dsos_obj_list_res ITER_END(dsos_container_id, dsos_iter_id) = 33;
        dsos_obj_list_res ITER_NEXT(dsos_container_id, dsos_iter_id) = 34;
        dsos_obj_list_res ITER_PREV(dsos_container_id, dsos_iter_id) = 35;
        dsos_obj_list_res ITER_FIND(dsos_container_id, dsos_iter_id) = 36;
        dsos_obj_list_res ITER_FIND_GLB(dsos_container_id, dsos_iter_id) = 37;
        dsos_obj_list_res ITER_FIND_LUB(dsos_container_id, dsos_iter_id) = 38;
    } = 1;
} = 40009862;
