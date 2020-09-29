#ifndef _5GS_REGISTRATION_TYPE_H_
#define _5GS_REGISTRATION_TYPE_H_

#include <stdint.h>

namespace nas{

class _5GSRegistrationType{
public:
  _5GSRegistrationType();
  _5GSRegistrationType(bool is_for, uint8_t type);
  _5GSRegistrationType(uint8_t iei, bool is_for, uint8_t type);
  ~_5GSRegistrationType();
  int decodefrombuffer(uint8_t *buf, int len, bool is_option);
  int encode2buffer(uint8_t *buf, int len);

  void setFollowOnReq(const bool is);
  void setRegType(const uint8_t type);
  bool isFollowOnReq();
  uint8_t getRegType();
private:
  uint8_t iei:4;
  bool is_for;
  uint8_t reg_type:3; 
};





}








#endif
