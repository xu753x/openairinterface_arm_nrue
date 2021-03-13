file(GLOB lte_rrc_source ASN1_files/lte-rrc-15.6.0.gen/*.c)
add_library(ASN1_LTE_RRC_LIB ${lte_rrc_source})
target_include_directories(ASN1_LTE_RRC_LIB PUBLIC "ASN1_files/lte-rrc-15.6.0.gen/")
target_compile_options(ASN1_LTE_RRC_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
