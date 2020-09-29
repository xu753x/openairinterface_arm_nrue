#include "multipartparser.hpp"

#include <string>
#include <iostream>
using namespace std;
extern "C"{
  #include <string.h>
}

bool multipart_parser(string input, string &jsonData, string &n1sm, string &n2sm){
    std::size_t found = input.find("Content-Type");
    std::string boundary_str = input.substr(2, found - 4);

    //step 1. use multipartparser to decode the input
    multipartparser_callbacks_init(&g_callbacks);
    g_callbacks.on_body_begin = &on_body_begin;
    g_callbacks.on_part_begin = &on_part_begin;
    g_callbacks.on_header_field = &on_header_field;
    g_callbacks.on_header_value = &on_header_value;
    g_callbacks.on_headers_complete = &on_headers_complete;
    g_callbacks.on_data = &on_data;
    g_callbacks.on_part_end = &on_part_end;
    g_callbacks.on_body_end = &on_body_end;

    multipartparser parser = {};
    init_globals();
    multipartparser_init(&parser, reinterpret_cast<const char*>(boundary_str.c_str()));

    unsigned int str_len = input.length();
    unsigned char *data = (unsigned char*) malloc(str_len + 1);
    memset(data, 0, str_len + 1);
    memcpy((void*) data, (void*) input.c_str(), str_len);

    if ((multipartparser_execute(&parser, &g_callbacks,
                               reinterpret_cast<const char*>(data), str_len)
      != strlen(input.c_str())) or (!g_body_begin_called)) {
      //return;
      std::cout<<"multipartparser_execute"<<std::endl;
    }

    //at least 2 parts for Json data and N1 (+ N2)
    if (g_parts.size() < 2){
      return false;
    }

    part p0 = g_parts.front(); g_parts.pop_front(); jsonData = p0.body;
    part p1 = g_parts.front(); g_parts.pop_front(); n1sm = p1.body;
    part p2; bool is_ngap = false;
    if (g_parts.size() > 0) {
      p2 = g_parts.front(); g_parts.pop_front();
      is_ngap = true;
    }
    if(is_ngap) n2sm = p2.body;
    else n2sm = "null";
    return true;
}

