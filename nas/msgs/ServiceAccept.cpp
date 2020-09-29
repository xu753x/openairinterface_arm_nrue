#include "ServiceAccept.hpp"
#include "logger.hpp"
#include "3gpp_ts24501.hpp"
using namespace nas;

ServiceAccept::ServiceAccept(){
	plain_header = NULL;
	ie_PDU_session_status = NULL;
	ie_session_reactivation_result = NULL;
}

ServiceAccept::~ServiceAccept(){}

void ServiceAccept::setHeader(uint8_t security_header_type){
  plain_header = new NasMmPlainHeader();
  plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, SERVICE_ACCEPT);
}

void ServiceAccept::setPDU_session_status(uint16_t value){
  ie_PDU_session_status = new PDU_Session_Status(0x50, value);
}

void ServiceAccept::setPDU_session_reactivation_result(uint16_t value){
  ie_session_reactivation_result = new PDU_Session_Reactivation_Result(0x26, value);
}

int ServiceAccept::encode2buffer(uint8_t *buf, int len){
  if(!plain_header){ 
    Logger::nas_mm().error("Missing message header");
    return -1;
  }
  int encoded_size = 0;
  if(!(plain_header->encode2buffer(buf, len))) return -1;
  encoded_size += 3;
  if(ie_PDU_session_status)
    encoded_size += ie_PDU_session_status->encode2buffer(buf+encoded_size, len-encoded_size);
  if(ie_session_reactivation_result)
    encoded_size += ie_session_reactivation_result->encode2buffer(buf+encoded_size, len-encoded_size);
  return encoded_size;
}
