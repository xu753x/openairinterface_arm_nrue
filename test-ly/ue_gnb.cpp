#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <unistd.h>  
#include <sys/socket.h>  
#include <sys/types.h>  
#include <netinet/in.h>  
#include <netinet/sctp.h>  
#include <arpa/inet.h>  
#include <iostream>
#include "sctp_server.hpp"
#include "ngap_app.hpp"

using namespace std; 
using namespace sctp;
using namespace ngap;
extern "C" int ue_gnb_simulator();
int encode_ng_setup_request(uint8_t*);
int encode_initial_ue_message(uint8_t*);
void handle_ngap_message(int sd, uint8_t*buf, int len);
void handle_authentication_request(uint8_t *nas, size_t nas_len, uint8_t *resStar);
extern void print_buffer(const std::string app, const std::string commit, uint8_t *buf, int len);

int ue_gnb_simulator(){
  printf("welcome to ue_gnb_simulator~~~~~~~~~~~~~~~~~~~~~~~~~\n");
  string  address = "192.168.17.1";
  uint16_t port_num = 38412;
  ngap_app NNgap(address,port_num);
  return 0;
}
