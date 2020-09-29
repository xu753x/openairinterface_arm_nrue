#ifndef _3GPP_TS_24501_H_
#define _3GPP_TS_24501_H_


/********** epd type **************/

#define EPD_5GS_MM_MSG  0b01111110
#define EPD_5GS_SM_MSG  0b00101110

/********** security header type **************/

#define PLAIN_5GS_MSG   0b0000
#define INTEGRITY_PROTECTED 0b0001
#define INTEGRITY_PROTECTED_AND_CIPHERED 0b0010
#define INTEGRITY_PROTECTED_WITH_NEW_SECU_CTX 0b0011                      // only for SECURITY MODE COMMAND
#define INTEGRITY_PROTECTED_AND_CIPHERED_WITH_NEW_SECU_CTX 0b0100         // only for SECURITY MODE COMPLETE

/********** message type ********************/

#define REGISTRATION_REQUEST  0b01000001
#define REGISTRATION_REJECT   0b01000100
#define AUTHENTICATION_REQUEST 0b01010110
#define AUTHENTICATION_RESPONSE 0b01010111
#define AUTHENTICATION_RESULT 0b01011010
#define AUTHENTICATION_FAILURE 0b01011001
#define AUTHENTICATION_REJECT  0b01011000
#define SECURITY_MODE_COMMAND  0b01011101
#define SECURITY_MODE_COMPLETE  0b01011110
#define SECURITY_MODE_REJECT    0b01011111
#define REGISTRATION_COMPLETE    0b01000011
#define REGISTRATION_ACCEPT     0b01000010
#define IDENTITY_REQUEST        0b01011011
#define IDENTITY_RESPONSE       0b01011100
#define UL_NAS_TRANSPORT        0b01100111
#define DL_NAS_TRANSPORT        0b01101000
#define SERVICE_REQUEST         0b01001100
#define SERVICE_REJECT          0b01001101
#define SERVICE_ACCEPT          0b01001110
#define UE_INIT_DEREGISTER      0b01000101

/********* registration type ***************/

#define NO_FOLLOW_ON_REQ_PENDING false
#define FOLLOW_ON_REQ_PENDING true

#define INITIAL_REGISTRATION 0b001
#define MOBILITY_REGISTRATION_UPDATING 0b010
#define PERIODIC_REGISTRATION_UPDATING 0b011
#define EMERGENCY_REGISTRATION 0b100

/********* NAS Key Set identifier **********/

#define NAS_KEY_SET_IDENTIFIER_NATIVE 0b0
#define NAS_KEY_SET_IDENTIFIER_MAPPED 0b1
#define NAS_KEY_SET_IDENTIFIER_NOT_AVAILABLE 0b111

/********* 5GSMobilityIdentity **********/

#define NO_IDENTITY 0b000
#define SUCI        0b001
#define _5G_GUTI    0b010
#define IMEI        0b011
#define _5G_S_TMSI  0b100
#define IMEISVI     0b101
#define MAC_ADDRESS 0b110

#define EVEN_IENTITY 0
#define ODD_IDENTITY 1

#define SUPI_FORMAT_IMSI 0b000
#define SUPI_FORMAT_NETWORK_SPECIFIC_IDENTIFIER 0b001

#define NULL_SCHEME 0b0000
#define ECIES_SCHEME_PROFILE_A 0b0001
#define ECIES_SCHEME_PROFILE_B 0b0010

#define HOME_NETWORK_PKI_0_WHEN_PSI_0 0b00000000
#define HOME_NETWORK_PKI_RESERVED     0b11111111

/************************** security algorithms ***********************************************************/
#define IA0_5G 0b000
#define IA1_128_5G 0b001
#define IA2_128_5G 0b010

#define EA0_5G 0b000
#define EA1_128_5G 0b001
#define EA2_128_5G 0b010

/************************** cause value for 5g mobility management(Annex A) ********************************/
#define _5GMM_CAUSE_ILLEGAL_UE 3
#define _5GMM_CAUSE_SYNCH_FAILURE 0b00010101


// A.5 Causes related to invalid messages
//------------------------------------------------------------------------------
#define _5GMM_CAUSE_SEMANTICALLY_INCORRECT                95
#define _5GMM_CAUSE_INVALID_MANDATORY_INFO                96
#define _5GMM_CAUSE_MESSAGE_TYPE_NOT_IMPLEMENTED          97
#define _5GMM_CAUSE_MESSAGE_TYPE_NOT_COMPATIBLE           98
#define _5GMM_CAUSE_IE_NOT_IMPLEMENTED                    99
#define _5GMM_CAUSE_CONDITIONAL_IE_ERROR                 100
#define _5GMM_CAUSE_MESSAGE_NOT_COMPATIBLE               101
#define _5GMM_CAUSE_PROTOCOL_ERROR                       111

/********************* UL NAS TRANSPORT payload container type **************************/
#define N1_SM_INFORMATION 0x01
#define SMS_CONTAINER               0x02
#define LTE_POSITIONING_PROTOCOL 0x03
#define SOR_TRANSPARENT_CONTAINER 0x04
#define UE_POLICY_CONTAINER 0x05
#define UE_PARAMETERS_UPDATE 0x06
#define MULTIPLE_PAYLOADS 0x0f

#define PDU_SESSION_INITIAL_REQUEST 0b001
#define EXISTING_PDU_SESSION        0b010
#define PDU_SESSION_INITIAL_EMERGENCY_REQUEST 0b011
#define EXISTING_EMERGENCY_PDU_SESSION 0b100
#define PDU_SESSION_MODIFICATION_REQUEST 0b101
#define MA_PDU_REQUEST 0b110












#endif
