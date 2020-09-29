#include "RegistrationAccept.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"
#include "String2Value.hpp"

using namespace nas;

RegistrationAccept::RegistrationAccept() {
  Logger::nas_mm().debug("initiating class RegistrationAccept");
  plain_header = NULL;
  ie_5gs_registration_result = NULL;
  ie_5g_guti = NULL;
  ie_equivalent_plmns = NULL;
  ie_allowed_nssai = NULL;
  ie_rejected_nssai = NULL;
  ie_configured_nssai = NULL;
  ie_5gs_network_feature_support = NULL;
  ie_PDU_session_status = NULL;
  ie_pdu_session_reactivation_result = NULL;
  ie_pdu_session_reactivation_result_error_cause = NULL;
  ie_MICO_indicationl = NULL;
  ie_network_slicing_indication = NULL;
  ie_T3512_value = NULL;
  ie_Non_3GPP_de_registration_timer_value = NULL;
  ie_T3502_value = NULL;
  ie_sor_transparent_container = NULL;
  ie_eap_message = NULL;
  ie_nssai_inclusion_mode = NULL;
  ie_negotiated_drx_parameters = NULL;
  ie_non_3gpp_nw_policies = NULL;
  ie_eps_bearer_context_status = NULL;
  ie_extended_drx_parameters = NULL;
  ie_T3447_value = NULL;
  ie_T3448_value = NULL;
  ie_T3324_value = NULL;
  ie_ue_radio_capability_id = NULL;
  ie_pending_nssai = NULL;

}

RegistrationAccept::~RegistrationAccept() {}

void RegistrationAccept::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, REGISTRATION_ACCEPT);
}
void RegistrationAccept::set_5GS_Registration_Result(bool emergency, bool nssaa, bool sms, uint8_t value){
	ie_5gs_registration_result = new _5GS_Registration_Result(0x00, emergency, nssaa, sms, value);
}
void RegistrationAccept::setSUCI_SUPI_format_IMSI(const string mcc, const string mnc, const string routingInd, uint8_t protection_sch_id, const string msin) {
	if (protection_sch_id != NULL_SCHEME) {
		Logger::nas_mm().error("encoding suci and supi format for imsi error, please choose right interface");
		return;
	}
	else {
		ie_5g_guti = new _5GSMobilityIdentity(mcc, mnc, routingInd, protection_sch_id, msin);
		ie_5g_guti->setIEI(0x77);
	}
}
void RegistrationAccept::setSUCI_SUPI_format_IMSI(const string mcc, const string mnc, const string routingInd, uint8_t protection_sch_id, uint8_t hnpki, const string msin) {}
void RegistrationAccept::set5G_GUTI(const string mcc, const string mnc, const string amfRegionId, const string amfSetId, const string amfPointer, const uint32_t tmsi) {
  ie_5g_guti = new _5GSMobilityIdentity();
  int regionId = fromString<int>(amfRegionId);
  int setId = fromString<int>(amfSetId);
  int pointer = fromString<int>(amfPointer);
  cout<<"amfRegionID string: "<<amfRegionId.c_str() <<endl;
  cout<<"amfRegionId uint8_t: "<<regionId<<endl;
  ie_5g_guti->set5GGUTI(mcc, mnc, (uint8_t)regionId, (uint16_t)setId, (uint8_t)pointer, tmsi);
  ie_5g_guti->setIEI(0x77);
}
void RegistrationAccept::setIMEI_IMEISV() {}
void RegistrationAccept::set5G_S_TMSI() {}

void RegistrationAccept::setEquivalent_PLMNs(uint8_t MNC_MCC1, uint8_t MNC_MCC2, uint8_t MNC_MCC3){
	ie_equivalent_plmns = new PLMN_List(0x4A, MNC_MCC1, MNC_MCC2, MNC_MCC3);
}
void RegistrationAccept::setALLOWED_NSSAI(std::vector<struct SNSSAI_s> nssai) {
	ie_allowed_nssai = new NSSAI(0x15, nssai);
}
void RegistrationAccept::setRejected_NSSAI(uint8_t cause, uint8_t value) {
	ie_rejected_nssai = new Rejected_NSSAI(0x11,cause, value);
}
void RegistrationAccept::setCONFIGURED_NSSAI(std::vector<struct SNSSAI_s> nssai) {
	ie_configured_nssai = new NSSAI(0x31, nssai);
}
void RegistrationAccept::set_5GS_Network_Feature_Support(uint8_t value, uint8_t value2) {
	ie_5gs_network_feature_support = new _5GS_Network_Feature_Support(0x21, value, value2);
}
void RegistrationAccept::setPDU_session_status(uint16_t value) {
	ie_PDU_session_status = new PDU_Session_Status(0x50, value);
}
void RegistrationAccept::setPDU_session_reactivation_result(uint16_t value) {
	ie_pdu_session_reactivation_result = new PDU_Session_Reactivation_Result(0x26, value);
}
void RegistrationAccept::setPDU_session_reactivation_result_error_cause(uint8_t session_id, uint8_t value) {
	ie_pdu_session_reactivation_result_error_cause = new PDU_Session_Reactivation_Result_Error_Cause(0x72,session_id, value);
}
void RegistrationAccept::setMICO_Indication(bool sprti, bool raai) {
	ie_MICO_indicationl = new MICO_Indication(0x0B, sprti,raai);
}
void RegistrationAccept::setNetwork_Slicing_Indication(bool dcni, bool nssci) {
        ie_network_slicing_indication = new Network_Slicing_Indication(0x09, dcni,nssci);
}
void RegistrationAccept::setT3512_Value(uint8_t unit, uint8_t value) {
	ie_T3512_value = new GPRS_Timer_3(0x5E,unit, value);
}
void RegistrationAccept::setNon_3GPP_de_registration_timer_value(uint8_t value) {
	ie_Non_3GPP_de_registration_timer_value = new GPRS_Timer_2(0x5D, value);
}
void RegistrationAccept::setT3502_value(uint8_t value) {
	ie_T3502_value = new GPRS_Timer_2(0x16, value);
}
void RegistrationAccept::setSOR_Transparent_Container(uint8_t header, uint8_t *value) {
	ie_sor_transparent_container = new SOR_Transparent_Container(0x73, header, value);
}
void RegistrationAccept::setEAP_Message(bstring eap) {
	ie_eap_message = new EAP_Message(0x78, eap);
}
void RegistrationAccept::setNSSAI_Inclusion_Mode(uint8_t value) {
	ie_nssai_inclusion_mode = new NSSAI_Inclusion_Mode(0x0A,value);
}
void RegistrationAccept::set_5GS_DRX_arameters(uint8_t value) {
	ie_negotiated_drx_parameters = new _5GS_DRX_arameters(0x51, value);
}
void RegistrationAccept::setNon_3GPP_NW_Provided_Policies(uint8_t value) {
	ie_non_3gpp_nw_policies = new Non_3GPP_NW_Provided_Policies(0x0D,value);
}
void RegistrationAccept::setEPS_Bearer_Context_Status(uint16_t value) {
	ie_eps_bearer_context_status = new EPS_Bearer_Context_Status(0x60, value);
}
void RegistrationAccept::setExtended_DRX_Parameters(uint8_t paging_time, uint8_t value) {
	ie_extended_drx_parameters = new Extended_DRX_Parameters(0x6E, paging_time, value);
}
void RegistrationAccept::setT3447_Value(uint8_t unit, uint8_t value) {
	ie_T3447_value = new GPRS_Timer_3(0x6C,unit, value);
}
void RegistrationAccept::setT3448_Value(uint8_t unit, uint8_t value) {
	ie_T3448_value = new GPRS_Timer_3(0x6B, unit, value);
}
void RegistrationAccept::setT3324_Value(uint8_t unit, uint8_t value) {
	ie_T3324_value = new GPRS_Timer_3(0x6A, unit, value);
}
void RegistrationAccept::setUE_Radio_Capability_ID(uint8_t value) {
	ie_ue_radio_capability_id = new UE_Radio_Capability_ID(0x67,value);
}
void RegistrationAccept::setPending_NSSAI(std::vector<struct SNSSAI_s> nssai) {
        ie_pending_nssai = new NSSAI(0x39, nssai);
}

void RegistrationAccept::setTaiList(std::vector<p_tai_t> tai_list){
  ie_tai_list = new _5GSTrackingAreaIdList(0x54, tai_list);
}

int RegistrationAccept::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding RegistrationAccept message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
	if (!ie_5gs_registration_result) {
		Logger::nas_mm().warn("IE ie_5gs_registration_result is not avaliable");
	}
	else {
		if (int size = ie_5gs_registration_result->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_5gs_registration_result  error");
			return 0;
		}
	}
	if (!ie_5g_guti) {
		Logger::nas_mm().warn("IE ie_5g_guti is not avaliable");
	}
	else {
                int size = ie_5g_guti->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_5g_guti  error");
			return 0;
		}
	}
        if(!ie_tai_list){
          Logger::nas_mm().warn("IE ie_tai_list is not avaliable");
        }else{
          int size = ie_tai_list->encode2buffer(buf+encoded_size, len-encoded_size);
          if(size != -1){
            encoded_size += size;
          }else{
	    Logger::nas_mm().error("encoding ie_tai_list  error");
	    return 0;
          }
        }
	if (!ie_equivalent_plmns) {
	  Logger::nas_mm().warn("IE ie_equivalent_plmns is not avaliable");
 	 }
 	 else {
	  if (int size = ie_equivalent_plmns->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_equivalent_plmns  error");
		  return 0;
	  }
  	}
	if (!ie_allowed_nssai) {
		Logger::nas_mm().warn("IE ie_allowed_nssai is not avaliable");
	}
	else {
		if (int size = ie_allowed_nssai->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_allowed_nssai  error");
			return 0;
		}
	}
	if (!ie_rejected_nssai) {
		Logger::nas_mm().warn("IE ie_rejected_nssai is not avaliable");
	}
	else {
		if (int size=ie_rejected_nssai->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size+=size;
		}
		else {
			Logger::nas_mm().error("encoding ie_rejected_nssai  error");
		}
	}
	if (!ie_configured_nssai) {
		Logger::nas_mm().warn("IE ie_configured_nssai is not avaliable");
	}
	else {
		if (int size = ie_configured_nssai->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_configured_nssai  error");
			return 0;
		}
	}
	if (!ie_5gs_network_feature_support) {
		Logger::nas_mm().warn("IE ie_5gs_network_feature_support is not avaliable");
	}
	else {
		if (int size = ie_5gs_network_feature_support->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_5gs_network_feature_support  error");
			return 0;
		}
	}
	if (!ie_PDU_session_status) {
	  Logger::nas_mm().warn("IE ie_PDU_session_status is not avaliable");
	  }
	  else {
	  if (int size = ie_PDU_session_status->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_PDU_session_status  error");
		  return 0;
	  }
 	 }
	if (!ie_pdu_session_reactivation_result) {
			Logger::nas_mm().warn("IE ie_pdu_session_reactivation_result is not avaliable");
		}
		else {
			if (int size = ie_pdu_session_reactivation_result->encode2buffer(buf + encoded_size, len - encoded_size)) {
				encoded_size += size;
			}
			else {
				Logger::nas_mm().error("encoding ie_pdu_session_reactivation_result  error");
				return 0;
			}
		}
	if (!ie_pdu_session_reactivation_result_error_cause) {
			Logger::nas_mm().warn("IE ie_pdu_session_reactivation_result_error_cause is not avaliable");
		}
		else {
			if (int size = ie_pdu_session_reactivation_result_error_cause->encode2buffer(buf + encoded_size, len - encoded_size)) {
				encoded_size += size;
			}
			else {
				Logger::nas_mm().error("encoding ie_pdu_session_reactivation_result_error_cause  error");
				return 0;
			}
		}
	if (!ie_MICO_indicationl) {
	  Logger::nas_mm().warn("IE ie_MICO_indicationl is not avaliable");
  	}
  	else {
	  if (int size = ie_MICO_indicationl->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_MICO_indicationl  error");
		  return 0;
	  }
  	}
	if (!ie_network_slicing_indication) {
          Logger::nas_mm().warn("IE ie_network_slicing_indication is not avaliable");
  	}
 	 else {
          if (int size = ie_network_slicing_indication->encode2buffer(buf + encoded_size, len - encoded_size)) {
                  encoded_size += size;
          }
          else {
                  Logger::nas_mm().error("encoding ie_network_slicing_indication  error");
                  return 0;
          }
  	}
	if (!ie_T3512_value) {
		Logger::nas_mm().warn("IE ie_T3512_value is not avaliable");
	}
	else {
		if (int size = ie_T3512_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_T3512_value  error");
			return 0;
		}
	}
	if (!ie_Non_3GPP_de_registration_timer_value) {
          Logger::nas_mm().warn("IE ie_Non_3GPP_de_registration_timer_value is not avaliable");
 	 }
	  else {
          if (int size = ie_Non_3GPP_de_registration_timer_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
                  encoded_size += size;
          }
          else {
                  Logger::nas_mm().error("encoding ie_Non_3GPP_de_registration_timer_value  error");
                  return 0;
          }
  	}
	if (!ie_T3502_value) {
	  Logger::nas_mm().warn("IE ie_T3502_value is not avaliable");
  	}
	  else {
	  if (int size = ie_T3502_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_T3502_value  error");
		  return 0;
	  }
	}
	 if (!ie_sor_transparent_container) {
                Logger::nas_mm().warn("IE ie_sor_transparent_container is not avaliable");
        }
        else {
                if (int size = ie_sor_transparent_container->encode2buffer(buf + encoded_size, len - encoded_size)) {
                        encoded_size += size;
                }
                else {
                        Logger::nas_mm().error("encoding ie_sor_transparent_container  error");
                        return 0;
                }
        }
	if (!ie_eap_message) {
          Logger::nas_mm().warn("IE ie_eap_message is not avaliable");
  	}
	  else {
          if (int size = ie_eap_message->encode2buffer(buf + encoded_size, len - encoded_size)) {
                  encoded_size += size;
          }
          else {
                  Logger::nas_mm().error("encoding ie_eap_message  error");
                  return 0;
          }
	}
	if (!ie_nssai_inclusion_mode) {
			Logger::nas_mm().warn("IE ie_nssai_inclusion_mode is not avaliable");
		}
		else {
			if (int size = ie_nssai_inclusion_mode->encode2buffer(buf + encoded_size, len - encoded_size)) {
				encoded_size += size;
			}
			else {
				Logger::nas_mm().error("encoding ie_nssai_inclusion_mode  error");
				return 0;
			}
		}
	if (!ie_negotiated_drx_parameters) {
	  Logger::nas_mm().warn("IE ie_negotiated_drx_parameters is not avaliable");
  	}
	  else {
	  if (int size = ie_negotiated_drx_parameters->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_negotiated_drx_parameters  error");
		  return 0;
	  }
  	}
	if (!ie_non_3gpp_nw_policies) {
			Logger::nas_mm().warn("IE ie_non_3gpp_nw_policies is not avaliable");
		}
		else {
			if (int size = ie_non_3gpp_nw_policies->encode2buffer(buf + encoded_size, len - encoded_size)) {
				encoded_size += size;
			}
			else {
				Logger::nas_mm().error("encoding ie_non_3gpp_nw_policies  error");
				return 0;
			}
		}
	if (!ie_eps_bearer_context_status) {
	  Logger::nas_mm().warn("IE ie_eps_bearer_context_status is not avaliable");
  	}
	  else {
	  if (int size = ie_eps_bearer_context_status->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_eps_bearer_context_status  error");
		  return 0;
	  }
  	}
	if (!ie_extended_drx_parameters) {
	  Logger::nas_mm().warn("IE ie_extended_drx_parameters is not avaliable");
 	 }
 	 else {
	  if (int size = ie_extended_drx_parameters->encode2buffer(buf + encoded_size, len - encoded_size)) {
		  encoded_size += size;
	  }
	  else {
		  Logger::nas_mm().error("encoding ie_extended_drx_parameters  error");
		  return 0;
	  }
  	}
	if (!ie_T3447_value) {
		Logger::nas_mm().warn("IE ie_T3447_value is not avaliable");
	}
	else {
		if (int size = ie_T3447_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_T3447_value  error");
			return 0;
		}
	}
	if (!ie_T3448_value) {
		Logger::nas_mm().warn("IE ie_T3448_value is not avaliable");
	}
	else {
		if (int size = ie_T3448_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_T3448_value  error");
			return 0;
		}
	}
	if (!ie_T3324_value) {
		Logger::nas_mm().warn("IE ie_T3324_value is not avaliable");
	}
	else {
		if (int size = ie_T3324_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_T3324_value  error");
			return 0;
		}
	}
	if (!ie_ue_radio_capability_id) {
			Logger::nas_mm().warn("IE ie_ue_radio_capability_id is not avaliable");
		}
		else {
			if (int size = ie_ue_radio_capability_id->encode2buffer(buf + encoded_size, len - encoded_size)) {
				encoded_size += size;
			}
			else {
				Logger::nas_mm().error("encoding ie_ue_radio_capability_id  error");
				return 0;
			}
		}
	if (!ie_pending_nssai) {
		Logger::nas_mm().warn("IE ie_pending_nssai is not avaliable");
	}
	else {
		if (int size = ie_pending_nssai->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_pending_nssai  error");
			return 0;
		}
	}
#if 0
        if(!ie_tai_list){
          Logger::nas_mm().warn("IE ie_tai_list is not avaliable");
        }else{
          int size = ie_tai_list->encode2buffer(buf+encoded_size, len-encoded_size);
          if(size != -1){
            encoded_size += size;
          }else{
	    Logger::nas_mm().error("encoding ie_tai_list  error");
	    return 0;
          }
        }
#endif
	Logger::nas_mm().debug("encoded RegistrationAccept message len(%d)", encoded_size);
	return encoded_size;
}

int RegistrationAccept::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding RegistrationAccept message");
	int decoded_size = 3;
	plain_header = header;
	ie_5gs_registration_result = new _5GS_Registration_Result();
		decoded_size += ie_5gs_registration_result->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
		switch((octet&0xf0)>>4){
			case 0xB:{
			Logger::nas_mm().debug("decoding iei(0xB)");
		  	ie_MICO_indicationl = new MICO_Indication();
			decoded_size += ie_MICO_indicationl->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);
    			}break;
			case 0x9:{
			Logger::nas_mm().debug("decoding iei(0x9)");
                  	ie_network_slicing_indication = new Network_Slicing_Indication();
                 	decoded_size += ie_network_slicing_indication->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
                	octet = *(buf + decoded_size);
                 	Logger::nas_mm().debug("next iei(0x%x)", octet);
   			}break;
			case 0xA: {
				Logger::nas_mm().debug("decoding iei(0xA)");
				ie_nssai_inclusion_mode = new NSSAI_Inclusion_Mode();
				decoded_size += ie_nssai_inclusion_mode->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0xD: {
				Logger::nas_mm().debug("decoding iei(0xD)");
				ie_non_3gpp_nw_policies = new Non_3GPP_NW_Provided_Policies();
				decoded_size += ie_non_3gpp_nw_policies->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			}
		switch (octet) {
			case 0x77: {
				Logger::nas_mm().debug("decoding iei(0x77)");
				ie_5g_guti = new _5GSMobilityIdentity();
				decoded_size += ie_5g_guti->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x15: {
				Logger::nas_mm().debug("decoding iei(0x15)");
				ie_allowed_nssai = new NSSAI();
				decoded_size += ie_allowed_nssai->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x11: {
				Logger::nas_mm().debug("decoding iei(0x11)");
				ie_rejected_nssai = new Rejected_NSSAI();
				decoded_size += ie_rejected_nssai->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);		
				}break;
			case 0x31: {
				Logger::nas_mm().debug("decoding iei(0x31)");
				ie_configured_nssai = new NSSAI();
				decoded_size += ie_configured_nssai->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x21: {
				Logger::nas_mm().debug("decoding iei(0x21)");
				ie_5gs_network_feature_support = new _5GS_Network_Feature_Support();
				decoded_size += ie_5gs_network_feature_support->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x50:{
				Logger::nas_mm().debug("decoding iei(0x50)");
		 		 ie_PDU_session_status = new PDU_Session_Status();
		 		 decoded_size += ie_PDU_session_status->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		 		 octet = *(buf + decoded_size);
		  		Logger::nas_mm().debug("next iei(0x%x)", octet);
   			}break;
			case 0x26: {
				Logger::nas_mm().debug("decoding iei(0x26)");
				ie_pdu_session_reactivation_result = new PDU_Session_Reactivation_Result();
				decoded_size += ie_pdu_session_reactivation_result->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x72: {
				Logger::nas_mm().debug("decoding iei(0x72)");
				ie_pdu_session_reactivation_result_error_cause = new PDU_Session_Reactivation_Result_Error_Cause();
				decoded_size += ie_pdu_session_reactivation_result_error_cause->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x5E: {
				Logger::nas_mm().debug("decoding iei(0x5E)");
				ie_T3512_value = new GPRS_Timer_3();
				decoded_size += ie_T3512_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x5D: {
				Logger::nas_mm().debug("decoding iei(0x5D)");
				ie_Non_3GPP_de_registration_timer_value = new GPRS_Timer_2();
				decoded_size += ie_Non_3GPP_de_registration_timer_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);		
			}break;
			case 0x16: {
				Logger::nas_mm().debug("decoding iei(0x16)");
				ie_T3502_value = new GPRS_Timer_2();
				decoded_size += ie_T3502_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x73: {
				Logger::nas_mm().debug("decoding iei(0x73)");
				ie_sor_transparent_container = new SOR_Transparent_Container();
				decoded_size += ie_sor_transparent_container->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x78: {
				Logger::nas_mm().debug("decoding iei(0x78)");
				ie_eap_message = new EAP_Message();
				decoded_size += ie_eap_message->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x51:{
				Logger::nas_mm().debug("decoding iei(0x51)");
				ie_negotiated_drx_parameters = new _5GS_DRX_arameters();
				  decoded_size += ie_negotiated_drx_parameters->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				  octet = *(buf + decoded_size);
				  Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x60:{
				Logger::nas_mm().debug("decoding iei(0x60)");
				  ie_eps_bearer_context_status = new EPS_Bearer_Context_Status();
				  decoded_size += ie_eps_bearer_context_status->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				  octet = *(buf + decoded_size);
				  Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x6E:{
				Logger::nas_mm().debug("decoding iei(0x6E)");
				ie_extended_drx_parameters = new Extended_DRX_Parameters();
		 		 decoded_size += ie_extended_drx_parameters->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		 		 octet = *(buf + decoded_size);
		  		Logger::nas_mm().debug("next iei(0x%x)", octet);
   			}break;
			case 0x6C: {
				Logger::nas_mm().debug("decoding iei(0x6C)");
				ie_T3447_value = new GPRS_Timer_3();
				decoded_size += ie_T3447_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x6B: {
				Logger::nas_mm().debug("decoding iei(0x6B)");
				ie_T3448_value = new GPRS_Timer_3();
				decoded_size += ie_T3448_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x6A: {
				Logger::nas_mm().debug("decoding iei(0x6A)");
				ie_T3324_value = new GPRS_Timer_3();
				decoded_size += ie_T3324_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x67: {
				Logger::nas_mm().debug("decoding iei(0x67)");
				ie_ue_radio_capability_id = new UE_Radio_Capability_ID();
				decoded_size += ie_ue_radio_capability_id->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x39: {
				Logger::nas_mm().debug("decoding iei(0x39)");
				ie_pending_nssai = new NSSAI();
				decoded_size += ie_pending_nssai->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
				octet = *(buf + decoded_size);
				Logger::nas_mm().debug("next iei(0x%x)", octet);
			}break;
			case 0x4A:{
       				 Logger::nas_mm().debug("decoding iei(0x4A)");
				ie_equivalent_plmns = new PLMN_List();
        			decoded_size += ie_equivalent_plmns->decodefrombuffer(buf+decoded_size,len-decoded_size,true);
        			octet = *(buf+decoded_size);
        			Logger::nas_mm().debug("next iei(0x%x)",octet);
      			}break;
			}
	}
	Logger::nas_mm().debug("decoded RegistrationAccept message len(%d)", decoded_size);


}






