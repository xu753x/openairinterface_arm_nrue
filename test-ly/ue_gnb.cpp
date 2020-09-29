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
using namespace std; 
extern "C" int ue_gnb_simulator();
int encode_ng_setup_request(uint8_t*);
int encode_initial_ue_message(uint8_t*);
void handle_ngap_message(int sd, uint8_t*buf, int len);
void handle_authentication_request(uint8_t *nas, size_t nas_len, uint8_t *resStar);
extern void print_buffer(const std::string app, const std::string commit, uint8_t *buf, int len);

int ue_gnb_simulator(){
  printf("welcome to ue_gnb_simulator~~~~~~~~~~~~~~~~~~~~~~~~~");
  string amf_addr = "192.168.17.1";
  //string amf_addr = "10.103.238.78";
  string local_addr = "192.168.17.1";
  int sd = -1;
  struct sctp_initmsg                     init;
  struct sctp_event_subscribe             events;
  if ((sd = socket (AF_INET6, SOCK_STREAM, IPPROTO_SCTP)) < 0) {
    printf("Socket creation failed: %s\n", strerror (errno));
    return -1;
  }
  struct sockaddr *bindx_add_addr = (struct sockaddr*)calloc (1, sizeof (struct sockaddr));
  if (inet_pton (AF_INET, local_addr.c_str(), &((struct sockaddr_in *)&bindx_add_addr[0])->sin_addr.s_addr) != 1){
  } else {
    ((struct sockaddr_in *)bindx_add_addr)->sin_port = 0;
    bindx_add_addr->sa_family = AF_INET;
  }
  if (sctp_bindx (sd, bindx_add_addr, 1, SCTP_BINDX_ADD_ADDR) < 0) {
    printf("Socket bind failed: %s\n", strerror (errno));
    return -1;
  }
  memset ((void *)&init, 0, sizeof (struct sctp_initmsg));
  /*
   * Request a number of in/out streams
   */
  init.sinit_num_ostreams = 32;
  init.sinit_max_instreams = 32;
  init.sinit_max_attempts = 5;
  if (setsockopt (sd, IPPROTO_SCTP, SCTP_INITMSG, &init, (socklen_t) sizeof (struct sctp_initmsg)) < 0) {
    printf("Setsockopt IPPROTO_SCTP_INITMSG failed: %s\n", strerror (errno));
    return -1;
  }
  memset ((void *)&events, 1, sizeof (struct sctp_event_subscribe));

  if (setsockopt (sd, IPPROTO_SCTP, SCTP_EVENTS, &events, sizeof (struct sctp_event_subscribe)) < 0) {
    printf("Setsockopt IPPROTO_SCTP_EVENTS failed: %s\n", strerror (errno));
    return -1;
  }

  struct sockaddr_in                      addr;

  memset (&addr, 0, sizeof (struct sockaddr_in));

  if (inet_pton (AF_INET, amf_addr.c_str(), &addr.sin_addr.s_addr) != 1) {
    printf("Failed to convert ip address %s to network type\n", amf_addr.c_str());
  }

  addr.sin_family = AF_INET;
  addr.sin_port = htons (38412);
  printf("[%d] Sending explicit connect to %s:%u\n", sd, amf_addr.c_str(), 38412);
    /*
     * Connect to remote host and port
     */
  if (sctp_connectx (sd, (struct sockaddr *)&addr, 1, NULL) < 0) {
    printf("Connect to %s:%u failed: %s\n", amf_addr.c_str(), 38412, strerror (errno));
  }
  uint8_t buffer[1000];
  int encoded_size = encode_ng_setup_request(buffer);
  sctp_sendmsg (sd, (const void *)buffer, encoded_size, NULL, 0, htonl (60), 0, 0, 0, 0);
  
  while(true){
    int  flags = 0, recvSize = 0;
    socklen_t                                                                 from_len = 0;
    struct sctp_sndrcvinfo                                      sinfo = {0};
    struct sockaddr_in                                      addr = {0};
    uint8_t                                                                    recvBuffer[4096] = {0};

    memset ((void *)&addr, 0, sizeof (struct sockaddr_in));
    from_len = (socklen_t) sizeof (struct sockaddr_in);
    memset ((void *)&sinfo, 0, sizeof (struct sctp_sndrcvinfo));
    recvSize = sctp_recvmsg (sd, (void *)recvBuffer, 4096, (struct sockaddr *)&addr, &from_len, &sinfo, &flags);
    if (flags & MSG_NOTIFICATION){
      union sctp_notification  *snp = (union sctp_notification *)recvBuffer;
      switch (snp->sn_header.sn_type){
        case SCTP_SHUTDOWN_EVENT:{
        }break;
        case SCTP_ASSOC_CHANGE:{
        }break;
        default:{
        }break;
      }
    }else{
      handle_ngap_message(sd, recvBuffer, recvSize);
    }
  }

}
#include "NGSetupRequest.hpp"
using namespace ngap;
int encode_ng_setup_request(uint8_t * buf){
  NGSetupRequestMsg * ng = new NGSetupRequestMsg();
  ng->setMessageType();
  ng->setGlobalRanNodeID("110", "01", Ngap_GlobalRANNodeID_PR_globalGNB_ID, 0x00000001);
  ng->setRanNodeName("bupt gnb");
  std::vector<struct SupportedItem_s> list;
  struct SupportedItem_s item;
  item.tac = 100;
  PlmnSliceSupport_t plmn;
  plmn.mcc = "110";
  plmn.mnc = "11";
  SliceSupportItem_t slice;
  slice.sst = "1";
  slice.sd = "1";
  plmn.slice_list.push_back(slice);
  item.b_plmn_list.push_back(plmn);
  list.push_back(item);
  ng->setSupportedTAList(list);
  ng->setDefaultPagingDRX(Ngap_PagingDRX_v32);
  return ng->encode2buffer(buf, 1000);
}

#include "InitialUEMessage.hpp"
int encode_initial_ue_message(uint8_t * buf){
  printf("Sending InitialUEMessage after receive NGsetupResponse");
  InitialUEMessageMsg * init = new InitialUEMessageMsg();
  init->setMessageType();
  init->setRanUENgapID(0x00000001);
  struct NrCgi_s cgi;
  cgi.mcc = "110";
  cgi.mnc = "01";
  cgi.nrCellID = 0x1;
  struct Tai_s tai;
  tai.mcc = "110";
  tai.mnc = "01";
  tai.tac = 100;
  init->setUserLocationInfoNR(cgi, tai);
  init->setRRCEstablishmentCause(Ngap_RRCEstablishmentCause_mo_Signalling);
  uint8_t reg[35] = {0x7e, 0x00, 0x41, 0x79, 0x00, 0x0d, 0x01, 0x64, 0xf0, 0x11, 0x00, 0x00, 0x00, 0x00, 0x10, 0x32, 0x54, 0x76, 0x98, 0x10, 0x01, 0x03, 0x2e, 0x02, 0xf0, 0xf0, 0x17, 0x07, 0xf0, 0xf0, 0xc0, 0x40, 0x01, 0x80, 0x30};
  init->setNasPdu(reg, 35);
  init->setUeContextRequest(Ngap_UEContextRequest_requested);
  return init->encode2buffer(buf, 2020);
}

extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_InitiatingMessage.h"
}
#include "DownLinkNasTransport.hpp"
#include "UplinkNASTransport.hpp"
#include "nas_algorithms.hpp"
uint8_t Knas_int[16];
void handle_ngap_message(int sd, uint8_t*buf, int len){
  Ngap_NGAP_PDU_t *ngap_msg_pdu = (Ngap_NGAP_PDU_t*)calloc(1,sizeof(Ngap_NGAP_PDU_t));
  asn_dec_rval_t rc = asn_decode(NULL,ATS_ALIGNED_CANONICAL_PER,&asn_DEF_Ngap_NGAP_PDU,(void**)&ngap_msg_pdu, buf , len);
  switch(ngap_msg_pdu->present){
    case Ngap_NGAP_PDU_PR_initiatingMessage:{
      switch(ngap_msg_pdu->choice.initiatingMessage->procedureCode){
        case Ngap_ProcedureCode_id_DownlinkNASTransport:{
          printf("recv DOWNLINK-NAS-TRANSPORT message\n");
          DownLinkNasTransportMsg *dn = new DownLinkNasTransportMsg();
          dn->decodefrompdu(ngap_msg_pdu);
          uint8_t *nas; size_t nas_len = 0;
          dn->getNasPdu(nas, nas_len);
          switch(nas[1]){
            case 0x00:{
              printf("plain message\n");
              switch(nas[2]){
                case 0x56:{
                  printf("recv AUTHENTICATION-REQUEST message\n");
                  uint8_t uplink[1000]; uint8_t resStar[16];
 //                 handle_authentication_request(nas, nas_len, resStar);
                  UplinkNASTransportMsg *up = new UplinkNASTransportMsg();
                  up->setMessageType();
                  up->setAmfUeNgapId(0x1);
                  up->setRanUeNgapId(0x1);
                  struct NrCgi_s cgi;
                  cgi.mcc = "110";
                  cgi.mnc = "01";
                  cgi.nrCellID = 0x1;
                  struct Tai_s tai;
                  tai.mcc = "110";
                  tai.mnc = "01";
                  tai.tac = 100;
                  up->setUserLocationInfoNR(cgi, tai);
                  uint8_t nas[21] = {0x7e, 0x00, 0x57, 0x2d, 0x10};
                  memcpy(&nas[5], resStar, 16);
                  up->setNasPdu(nas, 21);
                  int encoded_size = up->encode2buffer(uplink, 1000);
                  sctp_sendmsg (sd, (const void *)uplink, encoded_size, NULL, 0, htonl (60), 0, 0, 0, 0);
                }
              }
            }break;
            case 0x03:{
              printf("integrity protected message(new security)\n");
              uint32_t mac32 = ntohl(*(uint32_t*)&nas[2]);
              printf("received mac32 0x%x\n", mac32);
              uint8_t *message = &nas[6];
              uint32_t count = 0;
              nas_stream_cipher_t stream_cipher = {0};
              uint8_t mac[4];
              stream_cipher.key = Knas_int;
              stream_cipher.key_length = 16;
              stream_cipher.count = count;
              stream_cipher.bearer = 0x00;
              stream_cipher.direction = 1;
              stream_cipher.message = message;
              stream_cipher.blength = (nas_len - 6)*8;
              nas_algorithms::nas_stream_encrypt_nia2 (&stream_cipher, mac);
              
		//print_buffer("amf_n1", "calculated MAC", mac, 4);
            }break;
          }
        }break; 
      }break;
    }break;
    case Ngap_NGAP_PDU_PR_successfulOutcome:{
      switch(ngap_msg_pdu->choice.successfulOutcome->procedureCode){
        case Ngap_ProcedureCode_id_NGSetup:{
          printf("recv NG-SETUP-RESPONSE message\n");
          uint8_t buffer[2049];
          int encoded_size = encode_initial_ue_message(buffer);
          sctp_sendmsg (sd, (const void *)buffer, encoded_size, NULL, 0, htonl (60), 0, 0, 0, 0);
        }break;
      }
    }break;
  }
}
