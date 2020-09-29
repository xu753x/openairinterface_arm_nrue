#include "ServiceType.hpp"
using namespace nas;

ServiceType::ServiceType(){}

ServiceType::~ServiceType(){}

ServiceType::ServiceType(uint8_t iei, uint8_t stp){
  _iei = iei;
  value = stp;
}

int ServiceType::encode2buffer(uint8_t *buf, int len){
  if(len < 1) return -1;
  int encoded_size = 0;
  if(_iei){ *buf = _iei; encoded_size++; *(buf+encoded_size) = value; encoded_size++; return encoded_size;}
  else{ *buf = 0x00 | ((value & 0x0f)<<4); return 0;}
}

int ServiceType::decodefrombuffer(uint8_t *buf, int len, bool is_optional, bool is_high){
  if(len < 1) return -1;
  if(is_optional){
    _iei = *buf;
  }else{
    if(is_high) 
      value = ((*buf)&0xf0)>>4;
    else
      value = (*buf)&0x0f;
    return 0;
  }
}

uint8_t ServiceType::getValue(){
  return value;
}
