#!/bin/bash
GENERATED_FULL_DIR=$1
shift
ASN1_SOURCE_DIR=$1
shift
export ASN1C_PREFIX=$1
shift
options=$*
done_flag="$GENERATED_FULL_DIR"/done
if [ "$done_flag" -ot $ASN1_SOURCE_DIR ] ; then
   rm -f "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}*.c "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}*.h
   mkdir -p "$GENERATED_FULL_DIR"
   asn1c -pdu=all -fcompound-names -gen-PER -no-gen-OER -no-gen-example $options -D $GENERATED_FULL_DIR $ASN1_SOURCE_DIR |& egrep -v "^Copied|^Compiled" | sort -u
  if [ "$ASN1C_PREFIX" = "X2AP_" ] ; then
    sed -i 's/18446744073709551615))/18446744073709551615U))/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}E-RABUsageReport-Item.c
    sed -i 's/18446744073709551615 }/18446744073709551615U }/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}E-RABUsageReport-Item.c
  fi
  if [ "$ASN1C_PREFIX" = "S1AP_" ] ; then
    sed -i 's/18446744073709551615))/18446744073709551615U))/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}E-RABUsageReportItem.c
    sed -i 's/18446744073709551615 }/18446744073709551615U }/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}E-RABUsageReportItem.c
  fi
  if [ "$ASN1C_PREFIX" = "NGAP_" ] ; then
    sed -i 's/18446744073709551615))/18446744073709551615U))/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}VolumeTimedReport-Item.c
    sed -i 's/18446744073709551615 }/18446744073709551615U }/g' "$GENERATED_FULL_DIR"/${ASN1C_PREFIX}VolumeTimedReport-Item.c
  fi

fi
touch $done_flag
