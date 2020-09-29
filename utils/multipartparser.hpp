/*
 * https://github.com/iafonov/multipart-parser-c
 */
#ifndef MULTIPARTPARSER_H
#define MULTIPARTPARSER_H
#include <map>
#include <string>
#include <list>
extern "C" {

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
}
typedef struct multipartparser multipartparser;
typedef struct multipartparser_callbacks multipartparser_callbacks;

typedef int (*multipart_cb) (multipartparser*);
typedef int (*multipart_data_cb) (multipartparser*, const char* data, size_t size);

struct multipartparser {
    /** PRIVATE **/
    char        boundary[70];
    int         boundary_length;
    int         index;
    uint16_t    state;

    /** PUBLIC **/
    void* data;
};

struct multipartparser_callbacks {
    multipart_cb      on_body_begin;
    multipart_cb      on_part_begin;
    multipart_data_cb on_header_field;
    multipart_data_cb on_header_value;
    multipart_cb      on_headers_complete;
    multipart_data_cb on_data;
    multipart_cb      on_part_end;
    multipart_cb      on_body_end;
};

void multipartparser_init(multipartparser* parser, const char* boundary);

void multipartparser_callbacks_init(multipartparser_callbacks* callbacks);

size_t multipartparser_execute(multipartparser* parser,
                               multipartparser_callbacks* callbacks,
                               const char* data,
                               size_t size);

#define BOUNDARY "----Boundary"

typedef struct part {
  std::map<std::string,std::string> headers;
  std::string body;
} part;

static multipartparser_callbacks g_callbacks;

static bool             g_body_begin_called;
static std::string      g_header_name;
static std::string      g_header_value;
static std::list<part>  g_parts;
static bool             g_body_end_called;

static void init_globals()
{
  g_body_begin_called = false;
  g_header_name.clear();
  g_header_value.clear();
  g_parts.clear();
  g_body_end_called = false;
}

static int on_body_begin(multipartparser* /*parser*/)
{
  g_body_begin_called = true;
  return 0;
}

static int on_part_begin(multipartparser* /*parser*/)
{
  g_parts.push_back(part());
  return 0;
}

static void on_header_done()
{
  g_parts.back().headers[g_header_name] = g_header_value;
  g_header_name.clear();
  g_header_value.clear();
}

static int on_header_field(multipartparser* /*parser*/, const char* data, size_t size)
{
  if (g_header_value.size() > 0)
    on_header_done();
  g_header_name.append(data, size);
  return 0;
}

static int on_header_value(multipartparser* /*parser*/, const char* data, size_t size)
{
  g_header_value.append(data, size);
  return 0;
}

static int on_headers_complete(multipartparser* /*parser*/)
{
  if (g_header_value.size() > 0)
    on_header_done();
  return 0;
}

static int on_data(multipartparser* /*parser*/, const char* data, size_t size)
{

  std::string str;
  //g_parts.back().body.append(data, size);
  for(int i = 0;i < size; i++)
  {
    //printf("%02x ",data[i]);
    str.push_back(data[i]);
  }
  g_parts.back().body.append(str);

  return 0;
}

static int on_part_end(multipartparser* /*parser*/)
{
  return 0;
}

static int on_body_end(multipartparser* /*parser*/)
{
  g_body_end_called = true;
  return 0;
}

#endif // MULTIPARTPARSER_H
