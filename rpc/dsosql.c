/* -*- c-basic-offset: 8 -*- */
#define _GNU_SOURCE
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/queue.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/errno.h>
#include <regex.h>
#include <getopt.h>

#include <readline/readline.h>
#include <readline/history.h>

#include <pwd.h>

#include "dsos.h"
#include "dsosql.h"
#include "json.h"

typedef struct av_s {
	char *name;
	char *value_str;
	union sos_primary_u value;
	LIST_ENTRY(av_s) link;
} *av_t;

typedef struct av_list_s {
	int count;
	LIST_HEAD(av_list, av_s) head;
} *av_list_t;

typedef struct cmd_s *cmd_t;
typedef int (*cmd_fn_t)(cmd_t, av_list_t);

typedef struct cmd_arg_s {
	sos_type_t type;
	char *name;
} *cmd_arg_t;

struct cmd_s {
	char *name;     /* User printable name of the function. */
	cmd_fn_t cmd_fn;
	char *doc;      /* Documentation for this function.  */
	int args_count;
	struct cmd_arg_s args[255];   /* Array of valid arguments to the commmand */
};

int open_session(cmd_t, av_list_t avl);
int open_container(cmd_t, av_list_t avl);
int create_schema(cmd_t, av_list_t avl);
int show_schema(cmd_t, av_list_t avl);
int show_command(cmd_t, av_list_t avl);
int quit_command(cmd_t, av_list_t avl);
int import_csv(cmd_t, av_list_t avl);
int help_command(cmd_t, av_list_t avl);
int select_command(cmd_t, av_list_t avl);

/* A structure which contains information on the commands this program
   can understand. */

enum dsosql_command_id_e {
	ATTACH_CMD,
	OPEN_CMD,
	CREATE_SCHEMA_CMD,
	SHOW_SCHEMA_CMD,
	IMPORT_CMD,
	SELECT_CMD,
	SHOW_CMD,
	HELP_CMD,
	HELP_CMD_2,
	LAST_CMD,
};

struct cmd_s commands[] = {
	[ATTACH_CMD] = {"attach", open_session, "Open a session with the DSOSD cluster specified by FILE",
		1,
		{
			{ SOS_TYPE_STRING, "path" },
		}
	},
	[OPEN_CMD] = {"open", open_container, "open path PATH [perm SOS_PERM_RW/RO] [mode 0660]",
	      3,
	      {
		      { SOS_TYPE_STRING, "path"},
		      { SOS_TYPE_STRING, "perm"},
		      { SOS_TYPE_UINT32, "mode"}
	      }
	},
	[CREATE_SCHEMA_CMD] = { "create_schema", create_schema, "create_schema name SCHEMA-NAME from FILE-NAME",
		2,
		{
			 { SOS_TYPE_STRING, "name" },
			 { SOS_TYPE_STRING, "from" }
		}
	},
	[SHOW_SCHEMA_CMD] = { "show_schema", show_schema, "show_schema [ name NAME-RE ]",
		2,
		{
			{ SOS_TYPE_STRING, "name" },
			{ SOS_TYPE_STRING, "regex" },
		}
	},
	[IMPORT_CMD] = {"import", import_csv, "import schema SCHEMA-NAME from CSV-FILE-NAME",
		2,
		{
			{ SOS_TYPE_STRING, "schema" },
			{ SOS_TYPE_STRING, "from" }
		}
	},
	[SELECT_CMD] = {"select", select_command, "select COLS from SCHEMA where COND",
	},
	[SHOW_CMD] = {"show", show_command, "Display information about a DSOSD object.",
	      2,
	      {
		      { -1, "schemas" }
	      }
	},
	[HELP_CMD] = {"help", help_command, "help"},
	[HELP_CMD_2] = {"?", help_command, "Synonym for `help'"},
	[LAST_CMD] = {}
};

av_t av_find(av_list_t avl, const char *name)
{
	av_t av;
	LIST_FOREACH(av, &avl->head, link) {
		if (0 == strcmp(av->name, name))
			return av;
	}
	return NULL;
}

void av_free_args(av_list_t avl)
{
	av_t av;
	if (!avl)
		return;
	while (!LIST_EMPTY(&avl->head)) {
		av = LIST_FIRST(&avl->head);
		LIST_REMOVE(av, link);
		free(av->name);
		free(av->value_str);
		free(av);
	}
	free(avl);
}

av_list_t av_parse_args(cmd_t cmd, char *args_)
{
	char *args = strdup(args_);
	int arg_id;
	char *token, *value;
	av_list_t avlist = calloc(1, sizeof(*avlist));
	av_t av;
	/*
	 * If the command argument count is 0, it means that the args
	 * should be passed through unparsed
	 */
	if (cmd->args_count == 0) {
		av = calloc(1, sizeof(*av));
		av->name = strdup("");
		av->value_str = strdup(rl_line_buffer);
		LIST_INSERT_HEAD(&avlist->head, av, link);
		free(args);
		return avlist;
	}
	for (token = strtok(args, " ="); token && token[0] != '\0'; token = strtok(NULL, " =")) {
		/* Find this 'token' in the argument list */
		for (arg_id = 0; arg_id < cmd->args_count; arg_id++) {
			cmd_arg_t arg = &cmd->args[arg_id];
			if (strcmp(arg->name, token))
				continue;
			av = calloc(1, sizeof(*av));
			LIST_INSERT_HEAD(&avlist->head, av, link);
			avlist->count++;
			/* Parse the value string based on the stated type */
			av->name = strdup(token);
			if (arg->type == -1) {
				/* The token is the value */
				av->value_str = strdup(token);
				break;
			}
			value = strtok(NULL, " =");
			if (!value) {
				printf("The token '%s' expects a value.\n", token);
				goto err;
			}
			av->value_str = strdup(value);
			switch (arg->type) {
			case SOS_TYPE_INT16:
				av->value.int16_ = strtol(av->value_str, NULL, 0);
				break;
			case SOS_TYPE_INT32:
				av->value.int32_ = strtol(av->value_str, NULL, 0);
				break;
			case SOS_TYPE_INT64:
				av->value.int32_ = strtoll(av->value_str, NULL, 0);
				break;
			case SOS_TYPE_UINT16:
			case SOS_TYPE_UINT32:
			case SOS_TYPE_UINT64:
			case SOS_TYPE_FLOAT:
			default:
				break;
			}
		}
	}
	free(args);
	return avlist;
 err:
	free(args);
	av_free_args(avlist);
	return NULL;
}

/* Forward declarations. */
char *stripwhite();
cmd_t find_command();
static cmd_t current_command;

/* Execute a command line. */
int execute_line(char *line)
{
	register int i;
	cmd_t cmd;
	char *word;

	/* Isolate the command word. */
	i = 0;
	while (line[i] && whitespace(line[i]))
		i++;
	word = line + i;

	while (line[i] && !whitespace(line[i]))
		i++;

	if (line[i])
		line[i++] = '\0';

	cmd = find_command(word);
	if (!cmd) {
		fprintf(stderr, "%s: No such command for dsosql.\n", word);
		return (-1);
	}

	/* Get argument to command, if any. */
	while (whitespace(line[i]))
		i++;

	word = line + i;

	av_list_t avl = av_parse_args(cmd, word);
	int rc = cmd->cmd_fn(cmd, avl);
	av_free_args(avl);
	return rc;
}

cmd_t find_command(char *name)
{
	register int i;

	for (i = 0; commands[i].name; i++)
		if (strcmp(name, commands[i].name) == 0)
			return (&commands[i]);

	return ((cmd_t)NULL);
}

char *stripwhite(char *string)
{
	register char *s, *t;

	for (s = string; whitespace(*s); s++)
		;

	if (*s == 0)
		return (s);

	t = s + strlen(s) - 1;
	while (t > s && whitespace(*t))
		t--;
	*++t = '\0';

	return s;
}

char *command_generator();
char *argument_generator();
char **dsosql_completion();
extern char **completion_matches(char *text, void *entry_func);

/* Tell the GNU Readline library how to complete.  We want to try to complete
   on command names if this is the first word in the line, or on filenames
   if not. */
void initialize_readline()
{
	/* Allow conditional parsing of the ~/.inputrc file. */
	rl_readline_name = "dsosql";

	/* Tell the completer that we want a crack first. */
	rl_attempted_completion_function = dsosql_completion;
}

/* Attempt to complete on the contents of TEXT.  START and END show the
 * region of TEXT that contains the word to complete.  We can use the
 * entire line in case we want to do some simple parsing.  Return the
 * array of matches, or NULL if there aren't any. */
char **dsosql_completion(char *text, int start, int end)
{
	char **matches;

	matches = (char **)NULL;

	/*
	 * If this word is at the start of the line, then it is a command
	 * to complete.  Otherwise, an argument to the command.
	 */
	if (start == 0) {
		matches = completion_matches(text, command_generator);
	} else {
		char name[255];
		int i;
		for (i = 0; !isspace(rl_line_buffer[i]) && i < sizeof(name); i++)
			name[i] = rl_line_buffer[i];
		name[i] = 0;
		current_command = find_command(name);
		if (!current_command)
			goto out;
		matches = completion_matches(text, argument_generator);
	}
 out:
	return (matches);
}

/*
 * Generator function for command completion. If state is zero, we
 * reset the match index.
 */
char *command_generator(char *text, int state)
{
	static int list_index, len;
	char *name;

	/* If this is a new word to complete, initialize now.  This includes
	   saving the length of TEXT for efficiency, and initializing the index
	   variable to 0. */
	if (!state) {
		list_index = 0;
		len = strlen(text);
	}

	/* Return the next name which partially matches from the command list. */
	while (NULL != (name = commands[list_index].name)) {
		list_index++;
		if (strncmp(name, text, len) == 0)
			return (strdup(name));
	}

	/* If no names matched, then return NULL. */
	return ((char *)NULL);
}

char *argument_generator(char *text, int state)
{
	static int arg_index, len;
	char *name;

	if (current_command == &commands[ATTACH_CMD])
		return NULL;

	if (!state) {
		arg_index = 0;
		len = strlen(text);
	}

	/* Return the next name which partially matches from the command list. */
	while (current_command->args && NULL != (name = current_command->args[arg_index].name)) {
		arg_index++;
		if (strncmp(name, text, len) == 0)
			return (strdup(name));
	}

	/* If no names matched, then return NULL. */
	return NULL;
}

int help_command(cmd_t cmd, av_list_t avl)
{
	if (avl->count == 0) {
		int i, column = 0;
		for (i = 0; commands[i].name; i++) {
			if (column == 6) {
				column = 0;
				printf("\n");
			}
			printf("%s\t", commands[i].name);
			column++;
		}

		if (column)
			printf("\n");
	} else {
		av_t av = LIST_FIRST(&avl->head);
		cmd = find_command(av->name);
		if (cmd) {
			printf("usage: %s\n", cmd->doc);
		} else {
			printf("'%s' is not a dsosql command.\n", av->name);
		}
	}
	return 0;
}

static dsos_session_t g_session;
/* Connect to the cluster defined in the file argument. */
int open_session(cmd_t cmd, av_list_t avl)
{
	if (g_session) {
		printf("There is already an active session\n");
		return 0;
	}
	if (!avl->count) {
		printf("The '%s' command requires a FILE-NAME argument.\n", cmd->name);
		return 0;
	}
	av_t av = LIST_FIRST(&avl->head);
	g_session = dsos_session_open(av->value_str);
	if (!g_session) {
		printf("The cluster defined in '%s' is not available\n",
		       av->value_str);
		return 0;
	}
	return 0;
}

int show_command(cmd_t cmd, av_list_t avl)
{
	return 0;
}

dsos_container_t g_cont;
int open_container(cmd_t cmd, av_list_t avl)
{
	char *path;
	sos_perm_t perm;
	int mode;
	av_t av;

	if (!g_session) {
		printf("Please connect to a cluster with the 'attach' command "
		       "before attempting to open a container\n");
		return 0;
	}
	av = av_find(avl, "path");
	if (!av) {
		printf("The path parameter is required.\n");
		return 0;
	}
	path = av->value_str;
	av = av_find(avl, "perm");
	if (av) {
		if (0 == strcasecmp(av->value_str, "ro"))
			perm = SOS_PERM_RO;
		else
			perm = SOS_PERM_RW;
	} else {
		perm = SOS_PERM_RW;
	}
	av = av_find(avl, "mode");
	if (av) {
		mode = strtol(av->name, NULL, 0);
	} else {
		mode = 0660;
	}

	g_cont = dsos_container_open(g_session, path, perm, mode);
	if (!g_cont) {
		printf("Error %d opening the container.\n", errno);
	}
	return 0;
}

int show_schema(cmd_t cmd, av_list_t avl)
{
	dsos_res_t res;
	av_t av;

	if (!g_cont) {
		printf("You cannot query schema before you open a container\n");
		goto out;
	}

	av = av_find(avl, "name");
	if (!av) {
		av = av_find(avl, "regex");
		if (av)
			goto regex;
		dsos_name_array_t schemas = dsos_schema_query(g_cont, &res);
		if (schemas) {
			int i;
			char *name;
			for (i = 0; i < schemas->count; i++) {
				printf("%s\n", schemas->names[i]);
			}
		}
		goto out;
	} else {
		dsos_schema_t schema = dsos_schema_by_name(g_cont, av->value_str, &res);
		if (!schema) {
			printf("The schema '%s' could not b e found.\n", av->value_str);
			goto out;
		}
		sos_schema_print(dsos_schema_schema(schema), stdout);
	}

	av = av_find(avl, "regex");
 regex:
	if (!av) {
		printf("show_schema [name schema-name] [regex schema-name-re]\n");
		goto out;
	}

	regex_t rex;
	int rc = regcomp(&rex, av->value_str, REG_EXTENDED | REG_NOSUB);
	if (rc) {
		printf("The regular expression '%s' is invalid.\n", av->value_str);
		goto out_1;
	}

	dsos_name_array_t schemas = dsos_schema_query(g_cont, &res);
	if (schemas) {
		int i;
		char *name;
		for (i = 0; i < schemas->count; i++) {
			int rc = regexec(&rex, schemas->names[i], 0, NULL, 0);
			if (!rc) {
				dsos_schema_t schema = dsos_schema_by_name(g_cont, schemas->names[i], &res);
				if (schema)
					sos_schema_print(dsos_schema_schema(schema), stdout);
			}
		}
	}
 out_1:
	regfree(&rex);
 out:
	return 0;
}

int create_schema(cmd_t cmd, av_list_t avl)
{
	char *template;
	char *schema;
	av_t av;
	dsos_res_t res;
	size_t bytes;
	size_t read_size = 1024 * 1024;
	size_t buf_size = read_size;
	char *nbuf, *buf = malloc(read_size);
	if (!buf) {
		printf("Memory allocation failure.\n");
		return 0;
	}
	if (!g_cont) {
		printf("You cannot create a schema you open a container\n");
		goto err;
	}
	av = av_find(avl, "name");
	if (!av) {
		printf("The 'name' parameter is required.\n");
		goto err;
	}
	schema = av->value_str;
	av = av_find(avl, "from");
	if (!av) {
		printf("The 'from' parameter is required.\n");
		goto err;
	}
	template = av->value_str;
	FILE *fp = fopen(template, "ro");
	if (!fp) {
		printf("Error %d opening the file '%s'\n", errno, template);
		goto err;
	}
	nbuf = buf;
	size_t tot_bytes = 0;
	for (bytes = fread(nbuf, 1, read_size, fp);
	     bytes >= read_size;
	     bytes = fread(nbuf, 1, read_size, fp)) {
		tot_bytes += read_size;
		buf = realloc(buf, buf_size + read_size);
		if (!buf) {
			printf("Memory allocation failure.\n");
			goto err;
		}
		buf_size += read_size;
		nbuf = &buf[tot_bytes];
	}
	template = buf;
	int rc = dsosql_create_schema(g_cont, schema, template);
	free(buf);
	return 0;
 err:
	if (buf)
		free(buf);
	printf("usage: %s\n", cmd->doc);
	return 0;
}

int import_csv(cmd_t cmd, av_list_t avl)
{
	char *schema;
	char *path;
	av_t av;
	dsos_res_t res;

	if (!g_cont) {
		printf("You cannot import a CSV until you open a container\n");
		goto err;
	}
	av = av_find(avl, "schema");
	if (!av) {
		printf("The 'schema' parameter is required.\n");
		goto err;
	}
	schema = av->value_str;
	av = av_find(avl, "from");
	if (!av) {
		printf("The 'from' parameter is required.\n");
		goto err;
	}
	path = av->value_str;
	FILE *fp = fopen(path, "ro");
	if (!fp) {
		printf("Error %d opening the file '%s'\n", errno, path);
		goto err;
	}
	int rc = dsosql_import_csv(g_cont, fp, schema, NULL);
	return 0;
 err:
	printf("usage: %s\n", cmd->doc);
	return 0;
}

int select_command(cmd_t cmd, av_list_t avl)
{
	if (!g_cont) {
		printf("Use the 'open' command to open a container before using 'select'\n");
		return 0;
	}
	av_t av = LIST_FIRST(&avl->head);
	int rc = dsosql_query_select(g_cont, av->value_str);
	return 0;
}

static int done;
int quit_command(cmd_t cmd, av_list_t avl)
{
	done = 1;
	return (0);
}

static char *progname;
static char history_path[PATH_MAX];
void update_history(void)
{
	int rc = write_history(history_path);
	if (rc)
		printf("warning: write_history returned %d\n", rc);
}

void usage(int argc, char *argv[])
{
	fprintf(stderr, "dsosql: [-a attach-file -o open-file -h history-file]\n");
	fprintf(stderr, "        --attach  PATH The name of the cluster configuration file\n");
	fprintf(stderr, "        --open    NAME The name of the container to open\n");
	fprintf(stderr, "        --history PATH The desired location of the history file\n");
	exit(1);
}

static struct option long_opts[] = {
	{ "attach",	required_argument, 0, 'a' },
	{ "open",	required_argument, 0, 'o' },
	{ "history",	required_argument, 0, 'h' }
};

#define HISTORY_PATH	"DSOSQL_HISTORY_PATH"
#define HISTORY_FILE	"DSOSQL_HISTORY_FILE"

int main(int argc, char *argv[])
{
	char command[PATH_MAX];
	char *h_path;
	char *h_file;
	int rc;
	char *line, *s;
	struct passwd *pw = getpwuid(getuid());
	progname = argv[0];
	atexit(update_history);
	initialize_readline(); /* Bind our completer. */


	h_path = getenv(HISTORY_PATH);
	if (!h_path)
		h_path = pw->pw_dir;
	h_file = getenv(HISTORY_FILE);
	if (!h_file)
		h_file = ".dsosql_history";


	int opt_idx = 0;
	int opt;
	char *attach_file = NULL;
	char *open_file = NULL;
	while ((opt = getopt_long(argc, argv, "a:o:h:", long_opts, &opt_idx)) > 0) {
		switch (opt) {
		case 'a':
			attach_file = strdup(optarg);
			break;
		case 'o':
			open_file = strdup(optarg);
			break;
		case 'h':
			snprintf(history_path, sizeof(history_path), optarg);
			break;
		default:
			usage(argc, argv);
		}
	}
	if (history_path[0] == '\0')
		snprintf(history_path, sizeof(history_path), "%s/%s", h_path, h_file);
	rc = read_history(history_path);
	if (rc)
		printf("warning: read_history returned %d\n", rc);

	if (attach_file) {
		printf("Attaching to cluster %s ...", attach_file);
		fflush(stdout);
		snprintf(command, sizeof(command), "attach path %s", attach_file);
		execute_line(command);
		printf(" OK\n");
	}
	if (open_file) {
		if (!attach_file) {
			printf("The -o option must be specified with -a option\n");
			usage(argc, argv);
		}
		printf("Opening the container %s ...", open_file);
		fflush(stdout);
		snprintf(command, sizeof(command), "open path %s", open_file);
		execute_line(command);
		printf(" OK\n");
	}
	/* Loop reading and executing lines until the user quits. */
	for (; done == 0;) {
		line = readline("dsosql: ");
		if (!line)
			break;
		s = stripwhite(line);
		if (*s) {
			add_history(s);
			execute_line(s);
		}
		free(line);
	}
	exit(0);
}
