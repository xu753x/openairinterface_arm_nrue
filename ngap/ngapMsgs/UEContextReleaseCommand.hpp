#ifndef _UE_CONTEXT_RELEASE_COMMAND_H_
#define _UE_CONTEXT_RELEASE_COMMAND_H_

#include "MessageType.hpp"
#include "Cause.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"


extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_ProtocolIE-Field.h"
}


namespace ngap{

class UEContextReleaseCommandMsg{
public:
  UEContextReleaseCommandMsg();
  ~UEContextReleaseCommandMsg();
public:
  void setMessageType();
  void setAmfUeNgapId(unsigned long id);
  void setUeNgapIdPair(unsigned long amfId, uint32_t ranId);
  void addCauseIE();
  void setCauseRadioNetwork(e_Ngap_CauseRadioNetwork cause_value);
  void setCauseNas(e_Ngap_CauseNas cause_value);
  int encode2buffer(uint8_t *buf, int buf_size);
public:
  bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
private:
  Ngap_NGAP_PDU_t *pdu;
  Ngap_UEContextReleaseCommand_t *ies;

  AMF_UE_NGAP_ID *amfUeNgapId;
  RAN_UE_NGAP_ID *ranUeNgapId;
  Cause *causeValue;
};

}









#endif
