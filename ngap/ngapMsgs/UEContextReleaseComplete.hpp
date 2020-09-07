#ifndef _UE_CONTEXT_RELEASE_COMPLETE_H_
#define _UE_CONTEXT_RELEASE_COMPLETE_H_

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "NgapIEsStruct.hpp"
#include "UserLocationInformation.hpp"

extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_ProtocolIE-Field.h"
}


namespace ngap{

class UEContextReleaseCompleteMsg{
public:
  UEContextReleaseCompleteMsg();
  ~UEContextReleaseCompleteMsg();
public:
  void setMessageType();
  void setAmfUeNgapId(unsigned long id);
  void setRanUeNgapId(uint32_t ran_ue_ngap_id);
  void setUserLocationInfoNR(struct NrCgi_s cig, struct Tai_s tai);
  int  encode2buffer(uint8_t *buf, int buf_size);
private:
  Ngap_NGAP_PDU_t *pdu;
  Ngap_UEContextReleaseComplete_t *ies;
  AMF_UE_NGAP_ID *amfUeNgapId;
  RAN_UE_NGAP_ID *ranUeNgapId;
  UserLocationInformation *userLocationInformation;
};


}



















#endif
