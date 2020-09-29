#ifndef _NasKeySetIdentifier_H
#define _NasKeySetIdentifier_H

#include <stdint.h>

namespace nas{

class NasKeySetIdentifier{
public:
  NasKeySetIdentifier();
  NasKeySetIdentifier(const uint8_t iei, uint8_t tsc, uint8_t key_id);
  NasKeySetIdentifier(uint8_t tsc, uint8_t key_id);
  ~NasKeySetIdentifier();
  
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *buf, int len, bool is_option, bool is_high);

  void setTypeOfSecurityContext(uint8_t type);
  void setNasKeyIdentifier(uint8_t id);
  uint8_t getTypeOfSecurityContext();
  uint8_t getasKeyIdentifier();
private:
  uint8_t iei;
  uint8_t tsc;
  uint8_t key_id;
};





}






#endif
