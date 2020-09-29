#include "5GSRegistrationType.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

_5GSRegistrationType::_5GSRegistrationType(){
  iei = 0;
  is_for = false;
  reg_type = 0;
}

_5GSRegistrationType::_5GSRegistrationType(bool is_for, uint8_t type){
  this->is_for = is_for;
  this->reg_type = 0x07 & type;
  this->iei = 0;
}

_5GSRegistrationType::_5GSRegistrationType(uint8_t iei, bool is_for, uint8_t type){
  this->is_for = is_for;
  this->reg_type = 0x07 & type;
  this->iei = 0x0f & iei;
}

_5GSRegistrationType::~_5GSRegistrationType(){}

int _5GSRegistrationType::encode2buffer(uint8_t *buf, int len){
  Logger::nas_mm().error("encoding 5gsregistrationtype IE");
  if(len < 1) {
    Logger::nas_mm().error("encoding 5gsregistrationtype error(len is less than one)");
    return -1;
  }
  uint8_t octet = 0;
  if(is_for)
    octet = 0x08;
  octet |= reg_type;
  if(!(iei & 0x0f)){
    *buf = 0x0f & octet;
    Logger::nas_mm().debug("encoded 5GSRegistrationType IE(len(1/2 octet))");
    return 0;
  }else{
    *buf = (iei<<4) | octet;
    Logger::nas_mm().debug("encoded 5GSRegistrationType IE(len(1 octet))");
    return 1;
  }
}

int _5GSRegistrationType::decodefrombuffer(uint8_t *buf, int len, bool is_option){
  if(is_option){
    return -1;
  }
  Logger::nas_mm().debug("decoding 5GSRegistrationType");
  uint8_t octet = *buf;
  if(octet & 0x08)
    is_for = FOLLOW_ON_REQ_PENDING;
  else
    is_for = NO_FOLLOW_ON_REQ_PENDING;
  reg_type = 0x07 & octet;
  Logger::nas_mm().debug("decoded 5GSRegistrationType len(1/2 octet)");
  return 0;
}

void _5GSRegistrationType::setFollowOnReq(const bool is){
  is_for = is;
}

void _5GSRegistrationType::setRegType(const uint8_t type){
  reg_type = 0x07 & type; 
}

bool _5GSRegistrationType::isFollowOnReq(){
  return is_for;
}

uint8_t _5GSRegistrationType::getRegType(){
  return reg_type;
}
