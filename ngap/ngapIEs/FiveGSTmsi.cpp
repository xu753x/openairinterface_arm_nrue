#include "FiveGSTmsi.hpp"

using namespace ngap;

FiveGSTmsi::FiveGSTmsi(){}

FiveGSTmsi::~FiveGSTmsi(){}

bool FiveGSTmsi::decodefrompdu(Ngap_FiveG_S_TMSI_t pdu){
  amfSetid.decodefrombitstring(pdu.aMFSetID);
  amfPointer.decodefrombitstring(pdu.aMFPointer);
  uint32_t tmsi = ntohl(*(uint32_t*)pdu.fiveG_TMSI.buf);
  int size = pdu.fiveG_TMSI.size;
  std::string setId, pointer;
  amfSetid.getAMFSetID(setId);
  amfPointer.getAMFPointer(pointer);
  _5g_s_tmsi = setId+pointer+std::to_string(tmsi);
  return true;
}

void FiveGSTmsi::getValue(std::string &tmsi){
  tmsi = _5g_s_tmsi;
}
