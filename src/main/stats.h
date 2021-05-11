



#ifndef _EXECUTABLES_STATS_H_
#define _EXECUTABLES_STATS_H_

#include <forms.h>

/* Callbacks, globals and object handlers */

extern void reset_stats( FL_OBJECT *, long );

/* Forms and Objects */

typedef struct {
  FL_FORM    *stats_form;
  void       *vdata;
  char       *cdata;
  long        ldata;
  FL_OBJECT *stats_text;
  FL_OBJECT *stats_button;
} FD_stats_form;

extern FD_stats_form *create_form_stats_form( void );

#endif /* _EXECUTABLES_STATS_H_ */
