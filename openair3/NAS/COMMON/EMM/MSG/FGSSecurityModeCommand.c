/*! \file FGSSecurityModeCommand.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "FGSSecurityModeCommand.h"

int decode_fgs_security_mode_command(securityModeCommand_t *fgs_security_mode_com, uint8_t *buffer, uint32_t len)
{
  int decoded = 0;

  IES_DECODE_U8(buffer, decoded, fgs_security_mode_com->selectedNASsecurityalgorithms);

  fgs_security_mode_com->ngKSI = *(buffer + decoded) & 0x0f;
  decoded++;

  return decoded;

}