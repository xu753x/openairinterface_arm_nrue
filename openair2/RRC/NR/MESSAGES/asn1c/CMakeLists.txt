file(GLOB nr_rrc_source ASN1_files/nr-rrc-16.1.0.gen/*.c)
add_library(ASN1_NR_RRC_LIB ${nr_rrc_source})
target_include_directories(ASN1_NR_RRC_LIB PUBLIC "ASN1_files/nr-rrc-16.1.0.gen/")
target_compile_options(ASN1_NR_RRC_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT)
