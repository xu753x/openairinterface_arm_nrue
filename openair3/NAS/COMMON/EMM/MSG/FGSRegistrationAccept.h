#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "ExtendedProtocolDiscriminator.h"
#include "SecurityHeaderType.h"
#include "SpareHalfOctet.h"
#include "MessageType.h"
#include "FGSRegistrationResult.h"
#include "FGSMobileIdentity.h"

#include "NR_NAS_defs.h"
#include "nas_log.h"
#include "TLVDecoder.h"

#ifndef FGS_REGISTRATION_ACCEPT_H_
#define FGS_REGISTRATION_ACCEPT_H_

#define REGISTRATION_ACCEPT_MOBILE_IDENTITY                     0x77
#define REGISTRATION_ACCEPT_5GS_TRACKING_AREA_IDENTITY_LIST     0x54
#define REGISTRATION_ACCEPT_ALLOWED_NSSA                        0x15
#define REGISTRATION_ACCEPT_5GS_NETWORK_FEATURE_SUPPORT         0x21
#define REGISTRATION_ACCEPT_5GS_GPRS_timer3_T3512_value        0x5E

typedef struct FGSTrackingAreaIdentityList_tag {
  #define TRACKING_AREA_IDENTITY_LIST_ONE_PLMN_NON_CONSECUTIVE_TACS 0b00
  //#define TRACKING_AREA_IDENTITY_LIST_ONE_PLMN_NONCONSECUTIVE_TACS   0b01
  //#define TRACKING_AREA_IDENTITY_LIST_MANY_PLMNS      0b10
  uint8_t  typeoflist:2;
  uint8_t  numberofelements:5;
  uint8_t  mccdigit2:4;
  uint8_t  mccdigit1:4;
  uint8_t  mncdigit3:4;
  uint8_t  mccdigit3:4;
  uint8_t  mncdigit2:4;
  uint8_t  mncdigit1:4;
  unsigned int tac:24;
} FGSTrackingAreaIdentityList;

typedef struct NSSAI_tag {
  uint8_t length;
  uint8_t value;
} NSSAI;

typedef struct FGSNetworkFeatureSupport_tag {
  unsigned int mpsi:1;
  unsigned int iwkn26:1;
  unsigned int EMF:2;
  unsigned int EMC:2;
  unsigned int IMSVoPSN3GPP:1;
  unsigned int IMSVoPS3GPP:1;
  unsigned int MCSI:1;
  unsigned int EMCN3:1;
} FGSNetworkFeatureSupport;

typedef struct GPRStimer3_tag {
  uint8_t iei;
  uint8_t length;
  unsigned int unit:3;
  unsigned int timervalue:5;
} GPRStimer3;

typedef struct fgs_registration_accept_msg_tag {
    /* Mandatory fields */
    ExtendedProtocolDiscriminator           protocoldiscriminator;
    SecurityHeaderType                      securityheadertype:4;
    SpareHalfOctet                          sparehalfoctet:4;
    MessageType                             messagetype;
    FGSRegistrationResult                   fgsregistrationresult;
    FGSMobileIdentity                       fgsmobileidentity;
    FGSTrackingAreaIdentityList             tailist ;
    NSSAI                                   nssai;
    FGSNetworkFeatureSupport                fgsnetworkfeaturesupport;
    GPRStimer3                              gprstimer3;
} fgs_registration_accept_msg;

int decode_fgs_tracking_area_identity_list(FGSTrackingAreaIdentityList *tailist, uint8_t iei, uint8_t *buffer, uint32_t len);
int decode_fgs_allowed_nssa(NSSAI *nssai, uint8_t iei, uint8_t *buffer, uint32_t len);
int decode_fgs_network_feature_support(FGSNetworkFeatureSupport *fgsnetworkfeaturesupport, uint8_t iei, uint8_t *buffer, uint32_t len);
int decode_fgs_registration_accept(fgs_registration_accept_msg *fgs_registration_acc, uint8_t *buffer, uint32_t len);
int decode_fgs_gprs_timer3(GPRStimer3 *gprstimer3, uint8_t iei, uint8_t *buffer, uint32_t len);

#endif /* FGS REGISTRATION ACCEPT_H_*/