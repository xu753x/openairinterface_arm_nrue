file(GLOB lte_ngap_source R15/ngap-15.8.0.gen/*.c)
add_library(ASN1_NGAP_LIB ${lte_ngap_source})
target_include_directories(ASN1_NGAP_LIB PUBLIC "R15/ngap-15.8.0.gen/")
target_compile_options(ASN1_NGAP_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
