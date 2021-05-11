



#ifndef __SIMULATION_ETH_TRANSPORT_SOCKET__H__
#define __SIMULATION_ETH_TRANSPORT_SOCKET__H__

#    ifdef SOCKET_C
#        define private_socket(x) x
#        define public_socket(x) x
#    else
#        define private_socket(x)
#        define public_socket(x) extern x
#    endif
#    include "stdint.h"
public_socket (void socket_setnonblocking (int sockP);
              )
public_socket (int make_socket_inet (int typeP, uint16_t * portP, struct sockaddr_in *ptr_addressP);
              )
#endif
