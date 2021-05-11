#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "database.h"
#include "handler.h"
#include "config.h"
#include "logger/logger.h"
#include "view/view.h"
#include "gui/gui.h"

typedef struct {
  widget *pucch_pusch_iq_plot;
  logger *pucch_pusch_iq_logger;
} gnb_gui;

typedef struct {
  int socket;
  int *is_on;
  int nevents;
  pthread_mutex_t lock;
  gnb_gui *e;
  void *database;
} gnb_data;

void is_on_changed(void *_d)
{
  gnb_data *d = _d;
  char t;

  if (pthread_mutex_lock(&d->lock)) abort();

  if (d->socket == -1) goto no_connection;

  t = 1;
  if (socket_send(d->socket, &t, 1) == -1 ||
      socket_send(d->socket, &d->nevents, sizeof(int)) == -1 ||
      socket_send(d->socket, d->is_on, d->nevents * sizeof(int)) == -1)
    goto connection_dies;

no_connection:
  if (pthread_mutex_unlock(&d->lock)) abort();
  return;

connection_dies:
  close(d->socket);
  d->socket = -1;
  if (pthread_mutex_unlock(&d->lock)) abort();
}

void usage(void)
{
  printf(
"options:\n"
"    -d   <database file>      this option is mandatory\n"
"    -ip <host>                connect to given IP address (default %s)\n"
"    -p  <port>                connect to given port (default %d)\n",
  DEFAULT_REMOTE_IP,
  DEFAULT_REMOTE_PORT
  );
  exit(1);
}

static void *gui_thread(void *_g)
{
  gui *g = _g;
  gui_loop(g);
  return NULL;
}

static void gnb_main_gui(gnb_gui *e, gui *g, event_handler *h, void *database,
    gnb_data *ed)
{
  widget *main_window;
  widget *top_container;
  widget *line;
  widget *w;
  logger *l;
  view *v;

  main_window = new_toplevel_window(g, 500, 300, "gNB tracer");

  top_container = new_container(g, VERTICAL);
  widget_add_child(g, main_window, top_container, -1);

  line = new_container(g, HORIZONTAL);
  widget_add_child(g, top_container, line, -1);

  
  w = new_xy_plot(g, 55, 55, "", 50);
  e->pucch_pusch_iq_plot = w;
  widget_add_child(g, line, w, -1);
  xy_plot_set_range(g, w, -1000, 1000, -1000, 1000);
  xy_plot_set_title(g, w, "rxdataF");
  l = new_iqlog_full(h, database, "GNB_PHY_PUCCH_PUSCH_IQ", "rxdataF");
  v = new_view_xy(300*12*14,10,g,w,new_color(g,"#000"),XY_FORCED_MODE);
  logger_add_view(l, v);
  e->pucch_pusch_iq_logger = l;
}

int main(int n, char **v)
{
  char *database_filename = NULL;
  void *database;
  char *ip = DEFAULT_REMOTE_IP;
  int port = DEFAULT_REMOTE_PORT;
  int *is_on;
  int number_of_events;
  int i;
  event_handler *h;
  gnb_data gnb_data;
  gui *g;
  gnb_gui eg;

  for (i = 1; i < n; i++) {
    if (!strcmp(v[i], "-h") || !strcmp(v[i], "--help")) usage();
    if (!strcmp(v[i], "-d"))
      { if (i > n-2) usage(); database_filename = v[++i]; continue; }
    usage();
  }

  if (database_filename == NULL) {
    printf("ERROR: provide a database file (-d)\n");
    exit(1);
  }

  database = parse_database(database_filename);

  load_config_file(database_filename);

  number_of_events = number_of_ids(database);
  is_on = calloc(number_of_events, sizeof(int));
  if (is_on == NULL) abort();

  h = new_handler(database);

  on_off(database, "GNB_PHY_PUCCH_PUSCH_IQ", is_on, 1);

  gnb_data.database = database;
  gnb_data.socket = -1;
  gnb_data.is_on = is_on;
  gnb_data.nevents = number_of_events;
  if (pthread_mutex_init(&gnb_data.lock, NULL)) abort();

  g = gui_init();
  new_thread(gui_thread, g);

  gnb_main_gui(&eg, g, h, database, &gnb_data);

  OBUF ebuf = { osize: 0, omaxsize: 0, obuf: NULL };

restart:
  clear_remote_config();
  if (gnb_data.socket != -1) close(gnb_data.socket);
  gnb_data.socket = connect_to(ip, port);

  /* send the first message - activate selected traces */
  is_on_changed(&gnb_data);

  /* read messages */
  while (1) {
    event e;
    e = get_event(gnb_data.socket, &ebuf, database);
    if (e.type == -1) goto restart;
    if (pthread_mutex_lock(&gnb_data.lock)) abort();
    handle_event(h, e);
    if (pthread_mutex_unlock(&gnb_data.lock)) abort();
  }

  return 0;
}
