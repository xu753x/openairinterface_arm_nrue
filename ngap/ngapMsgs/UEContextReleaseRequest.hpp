#ifndef _UE_CONTEXT_RELEASE_REQUEST_H_
#define _UE_CONTEXT_RELEASE_REQUEST_H_

#include "MessageType.hpp"
#include "Cause.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"

extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_ProtocolIE-Field.h"
}

namespace ngap{

class UEContextReleaseRequestMsg{
public:
  UEContextReleaseRequestMsg();
  ~UEContextReleaseRequestMsg();
public:
  void setMessageType();
  void setAmfUeNgapId(unsigned long id);
  void setRanUeNgapId(uint32_t ran_ue_ngap_id);
  void addCauseIE();
  void setCauseRadioNetwork(e_Ngap_CauseRadioNetwork cause_value);
  int  encode2buffer(uint8_t *buf, int buf_size);
  bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
public:
  bool getCauseRadioNetwork(e_Ngap_CauseRadioNetwork &causeRadioNetwork);
  unsigned long getAmfUeNgapId();
  uint32_t getRanUeNgapId();
private:
  Ngap_NGAP_PDU_t *pdu;
  Ngap_UEContextReleaseRequest_t * ies;

  AMF_UE_NGAP_ID *amfUeNgapId;
  RAN_UE_NGAP_ID *ranUeNgapId;
  Cause *causeValue;
};


}
















#endif
