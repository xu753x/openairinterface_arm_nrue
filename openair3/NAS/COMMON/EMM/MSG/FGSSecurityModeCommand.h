#include "NR_NAS_defs.h"
#include "nas_log.h"
#include "TLVDecoder.h"

#ifndef FGS_SECURITY_MODE_COMMAND_H_
#define FGS_SECURITY_MODE_COMMAND_H_

int decode_fgs_security_mode_command(securityModeCommand_t *fgs_security_mode_com, uint8_t *buffer, uint32_t len);

#endif /* FGS SECURITY MODE COMMAND_H_*/