#ifndef _EPD_H_
#define _EPD_H_

#include<stdint.h>

namespace nas{

class ExtendedProtocolDiscriminator{
public:
  void encode2buffer(uint8_t *buf, int len);
  void setValue(const uint8_t epd);
  uint8_t getValue();
private:
  uint8_t m_epd; 

};











}







#endif
