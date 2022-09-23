# **********************************************************
 # Copyright (c) 2020 Xuhpclab. All rights reserved.
 # Licensed under the MIT License.
 # See LICENSE file for more information.
 #  **********************************************************

 cmake_minimum_required(VERSION 2.6)

 include(../../make/policies.cmake NO_POLICY_SCOPE)

 if (UNIX)
   add_compile_options(-Wno-deprecated)
   add_compile_options(-Wno-unused-result)
   add_compile_options(-Wno-unused-variable)
   add_compile_options(-Wno-unused-local-typedefs)
   add_compile_options(-Wno-unused-function)
   add_compile_options(-Werror=sign-compare)
   add_compile_options(-Werror=narrowing)
   add_compile_options(-std=c++11)

  if (SAMPLE_RUN)
    add_compile_options(-DSAMPLE_RUN)
  endif (SAMPLE_RUN)
   if (DEBUG)
     add_compile_options(-g3)
   endif (DEBUG)
 endif (UNIX)

 # add third-party libraries

 add_library(drcctlib_storespy SHARED
   drcctlib_storespy.cpp
   )

 configure_DynamoRIO_client(drcctlib_storespy)
 use_DynamoRIO_extension(drcctlib_storespy drcctlib)
 use_DynamoRIO_extension(drcctlib_storespy droption) 
 place_shared_lib_in_lib_dir(drcctlib_storespy)

 add_dependencies(drcctlib_storespy api_headers)

 # Provide a hint for how to use the client
 if (NOT DynamoRIO_INTERNAL OR NOT "${CMAKE_GENERATOR}" MATCHES "Ninja")
   add_custom_command(TARGET drcctlib_storespy
     POST_BUILD
     COMMAND ${CMAKE_COMMAND}
     ARGS -E echo "Usage: pass to drconfig or drrun: -t drcctlib_storespy"
     VERBATIM)
 endif ()

 install_target(drcctlib_storespy ${INSTALL_CLIENTS_LIB})

 set(INSTALL_CCTTEST_CONFIG ${INSTALL_CLIENTS_BASE})

 function (write_config_file dst bindir libdir)
   file(WRITE  ${dst} "# drcctlib_storespy tool config file\n")
   file(APPEND ${dst} "CLIENT_REL=${libdir}/${LIB_PFX}drcctlib_storespy${LIB_EXT}\n")
 endfunction ()

 if (X64)
   set(CONFIG_INSTALL ${PROJECT_BINARY_DIR}/drcctlib_storespy.drrun64)
   set(CONFIG_BUILD ${PROJECT_BINARY_DIR}/tools/drcctlib_storespy.drrun64)
 else (X64)
   set(CONFIG_INSTALL ${PROJECT_BINARY_DIR}/drcctlib_storespy.drrun32)
   set(CONFIG_BUILD ${PROJECT_BINARY_DIR}/tools/drcctlib_storespy.drrun32)
 endif (X64)

 set(BUILD_CLIENTS_BIN clients/${INSTALL_BIN})
 set(BUILD_CLIENTS_LIB clients/${INSTALL_LIB})

 write_config_file(${CONFIG_INSTALL} ${INSTALL_CLIENTS_BIN} ${INSTALL_CLIENTS_LIB})
 write_config_file(${CONFIG_BUILD} ${BUILD_CLIENTS_BIN} ${BUILD_CLIENTS_LIB})

 DR_install(FILES "${CONFIG_INSTALL}" DESTINATION ${INSTALL_CCTTEST_CONFIG})
 register_tool_file("drcctlib_storespy")

 ##################################################
 # Documentation