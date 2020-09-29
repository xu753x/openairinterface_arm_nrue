#ifndef _UESecurityCapability_H
#define _UESecurityCapability_H

#include <stdint.h>

namespace nas{

class UESecurityCapability{
public:
  UESecurityCapability(uint8_t iei);
  UESecurityCapability();
  ~UESecurityCapability();
  UESecurityCapability(const uint8_t iei, uint8_t _5gg_EASel, uint8_t _5gg_IASel);
  void setEASel(uint8_t sel);
  void setIASel(uint8_t sel);
  uint8_t getEASel();
  uint8_t getIASel();
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *buf, int len, bool is_option);
private:
  uint8_t _iei;
  uint8_t length;
  uint8_t _5g_EASel;
  uint8_t _5g_IASel;
};







}















#endif

