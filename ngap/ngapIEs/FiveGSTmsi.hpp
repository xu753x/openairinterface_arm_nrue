#ifndef _FiveGSTmsi_H_
#define _FiveGSTmsi_H_

#include "AMFSetID.hpp"
#include "AMFPointer.hpp"
#include <string>
extern "C"{
#include "Ngap_FiveG-S-TMSI.h"
}

namespace ngap{

class FiveGSTmsi{
public:
  FiveGSTmsi();
  ~FiveGSTmsi();
public:
  bool decodefrompdu(Ngap_FiveG_S_TMSI_t pdu);
  void getValue(std::string &value);
private:
  std::string _5g_s_tmsi;

  AMFSetID amfSetid;
  AMFPointer amfPointer;
};









}























#endif
