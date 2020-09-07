/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NGAP-IEs"
 * 	found in "asn.1/Information Element Definitions.asn1"
 * 	`asn1c -pdu=all -fcompound-names -fno-include-deps -findirect-choice -gen-PER -D src`
 */

#ifndef	_Ngap_SONInformation_H_
#define	_Ngap_SONInformation_H_


#include <asn_application.h>

/* Including external dependencies */
#include "Ngap_SONInformationRequest.h"
#include <constr_CHOICE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum Ngap_SONInformation_PR {
	Ngap_SONInformation_PR_NOTHING,	/* No components present */
	Ngap_SONInformation_PR_sONInformationRequest,
	Ngap_SONInformation_PR_sONInformationReply,
	Ngap_SONInformation_PR_choice_Extensions
} Ngap_SONInformation_PR;

/* Forward declarations */
struct Ngap_SONInformationReply;
struct Ngap_ProtocolIE_SingleContainer;

/* Ngap_SONInformation */
typedef struct Ngap_SONInformation {
	Ngap_SONInformation_PR present;
	union Ngap_SONInformation_u {
		Ngap_SONInformationRequest_t	 sONInformationRequest;
		struct Ngap_SONInformationReply	*sONInformationReply;
		struct Ngap_ProtocolIE_SingleContainer	*choice_Extensions;
	} choice;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} Ngap_SONInformation_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_Ngap_SONInformation;
extern asn_CHOICE_specifics_t asn_SPC_Ngap_SONInformation_specs_1;
extern asn_TYPE_member_t asn_MBR_Ngap_SONInformation_1[3];
extern asn_per_constraints_t asn_PER_type_Ngap_SONInformation_constr_1;

#ifdef __cplusplus
}
#endif

#endif	/* _Ngap_SONInformation_H_ */
#include <asn_internal.h>
