#ifndef _MessageType_H_
#define _MessageType_H_

extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_ProcedureCode.h"
  #include "Ngap_InitiatingMessage.h"
  #include "Ngap_SuccessfulOutcome.h"
  #include "Ngap_UnsuccessfulOutcome.h"
  #include "Ngap_Criticality.h"
}

namespace ngap{

class MessageType{
public:
  MessageType();
  virtual ~MessageType();

  int encode2pdu(Ngap_NGAP_PDU_t*&);
  void setProcedureCode(Ngap_ProcedureCode_t m_procedureCode);
  void setTypeOfMessage(Ngap_NGAP_PDU_PR m_typeOfMessage);
  void setCriticality(Ngap_Criticality m_criticality);
  void setValuePresent(Ngap_InitiatingMessage__value_PR m_valuePresent);
  void setValuePresent(Ngap_SuccessfulOutcome__value_PR m_valuePresent);
  void setValuePresent(Ngap_UnsuccessfulOutcome__value_PR m_valuePresent);

  //void decodefrompdu(Ngap_NGAP_PDU_t*);
  Ngap_ProcedureCode_t getProcedureCode();
  Ngap_NGAP_PDU_PR     getTypeOfMessage();
  Ngap_Criticality	   getCriticality();
private:
  Ngap_ProcedureCode_t procedureCode;//Ngap_ProcedureCode.h
  Ngap_NGAP_PDU_PR     typeOfMessage;//Ngap_NGAP-PDU.h
  Ngap_Criticality     criticality;//Ngap_Criticality.h
  Ngap_InitiatingMessage__value_PR initiatingMsgValuePresent;//Ngap_InitiatingMessage.h
  Ngap_SuccessfulOutcome__value_PR successfulOutcomeValuePresent;//Ngap_SuccessfulOutcome.h
  Ngap_UnsuccessfulOutcome__value_PR unsuccessfulOutcomeValuePresent;//Ngap_UnsuccessfulOutcome.h
};
}

#endif
