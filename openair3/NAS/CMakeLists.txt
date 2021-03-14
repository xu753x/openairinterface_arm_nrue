add_library(CONF2UEDATA_LIB
    TOOLS/conf_emm.c
    TOOLS/conf_parser.c
    TOOLS/conf_user_data.c
    TOOLS/conf_usim.c
    TOOLS/conf_network.c
    TOOLS/conf_user_plmn.c
    TOOLS/conf_parser.c
    TOOLS/fs.c
    TOOLS/display.c
    UE/API/USIM/usim_api.c
    UE/API/USIM/aka_functions.c
    COMMON/UTIL/memory.c
    COMMON/UTIL/OctetString.c
    COMMON/UTIL/TLVEncoder.c
)
target_link_libraries(CONF2UEDATA_LIB PRIVATE small-utils)
target_include_directories(CONF2UEDATA_LIB PRIVATE
    ../../openair2/COMMON
    COMMON/
    UE/API/USER/
    UE/API/USIM/
    UE/EMM/
    UE/ESM/
    COMMON/IES/
    COMMON/UTIL/
)

add_executable(conf2uedata TOOLS/conf2uedata.c)
target_link_libraries(conf2uedata PRIVATE CONF2UEDATA_LIB config)
# copies conf2uedata to top-level build directory for backwards compatibility
add_custom_command(
  TARGET conf2uedata POST_BUILD
  COMMAND cmake -E copy conf2uedata ${CMAKE_BINARY_DIR}/
  COMMENT "copy conf2uedata to top-level build directory"
  VERBATIM
)

add_executable(usim TOOLS/usim.c)
target_link_libraries(usim PRIVATE CONF2UEDATA_LIB config)
# copies usim to top-level build directory for backwards compatibility
add_custom_command(
  TARGET usim POST_BUILD
  COMMAND cmake -E copy usim ${CMAKE_BINARY_DIR}/
  COMMENT "copy usim to top-level build directory"
  VERBATIM
)

add_executable(nvram TOOLS/nvram)
target_link_libraries(nvram PRIVATE CONF2UEDATA_LIB config)
# copies nvram to top-level build directory for backwards compatibility
add_custom_command(
  TARGET nvram POST_BUILD
  COMMAND cmake -E copy nvram ${CMAKE_BINARY_DIR}/
  COMMENT "copy nvram to top-level build directory"
  VERBATIM
)
