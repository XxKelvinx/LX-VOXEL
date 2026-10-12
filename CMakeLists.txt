cmake_minimum_required(VERSION 3.16)
project(LX_VOXEL VERSION 0.1.5 LANGUAGES CXX)

# Release is the normal, reproducible configuration for a fresh checkout.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

option(LXVOXEL_NATIVE "Optimize for the build machine ISA" OFF)
option(LXVOXEL_BUILD_TESTS "Build supported platform integration tests" OFF)
option(LXVOXEL_BUILD_SHARED "Build reusable LX-VOXEL decoder library" ON)
option(LXVOXEL_SINGLE_THREAD "Build encoder for one-thread execution (no worker threads)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
find_package(Threads REQUIRED)

add_library(lxvoxel_internal INTERFACE)
target_include_directories(lxvoxel_internal INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(lxvoxel_internal INTERFACE Threads::Threads)
# Compile-time removal of old environment-tuned research pathways.
if(NOT MSVC)
  target_compile_options(lxvoxel_internal INTERFACE -include "${CMAKE_CURRENT_SOURCE_DIR}/src/release_defaults.hpp")
else()
  target_compile_options(lxvoxel_internal INTERFACE /FI"${CMAKE_CURRENT_SOURCE_DIR}/src/release_defaults.hpp")
endif()

add_library(lxvoxel_metadata STATIC src/metadata/metadata.cpp)
set_target_properties(lxvoxel_metadata PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lxvoxel_metadata PUBLIC lxvoxel_internal)

add_library(lxvoxel_audio_input STATIC src/io/audio_input.cpp)
target_link_libraries(lxvoxel_audio_input PUBLIC lxvoxel_internal)
if(WIN32)
  target_link_libraries(lxvoxel_audio_input PRIVATE mfplat mfreadwrite mfuuid ole32)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/windows/property-handler/lxvoxel_property_handler.cpp")
    add_library(lxvoxel-property-handler SHARED
      windows/property-handler/lxvoxel_property_handler.cpp)
  target_link_libraries(lxvoxel-property-handler PRIVATE propsys shlwapi ole32 windowscodecs gdi32)
    set_target_properties(lxvoxel-property-handler PROPERTIES PREFIX "")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_link_options(lxvoxel-property-handler PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
  endif()
endif()

# Production ISA set: x86 SSE4.1 baseline + AVX2/FMA fast path, or ARM NEON.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LXVOXEL_ARCH)
set(LXVOXEL_X86 OFF)
set(LXVOXEL_ARM OFF)
if(LXVOXEL_ARCH MATCHES "x86_64|amd64|i[3-6]86")
  set(LXVOXEL_X86 ON)
elseif(LXVOXEL_ARCH MATCHES "aarch64|arm64|^arm")
  set(LXVOXEL_ARM ON)
else()
  message(FATAL_ERROR "LX-VOXEL supports production builds on x86 (SSE4.1/AVX2+FMA) and ARM (NEON) only")
endif()

if(LXVOXEL_X86)
  add_library(lxvoxel_encoder_sse41 OBJECT src/encoder/main.cpp)
  target_link_libraries(lxvoxel_encoder_sse41 PRIVATE lxvoxel_internal)
  target_compile_definitions(lxvoxel_encoder_sse41 PRIVATE
    LXVOXEL_ENCODER_VARIANT=1
    LXVOXEL_ENCODER_ENTRY=lxvoxel_encoder_sse41_main)
  if(LXVOXEL_SINGLE_THREAD)
    target_compile_definitions(lxvoxel_encoder_sse41 PRIVATE LXVOXEL_SINGLE_THREAD=1)
  endif()
  lxvoxel_apply_sse41_optimizations(lxvoxel_encoder_sse41)

  add_library(lxvoxel_encoder_avx2 OBJECT src/encoder/main.cpp)
  target_link_libraries(lxvoxel_encoder_avx2 PRIVATE lxvoxel_internal)
  target_compile_definitions(lxvoxel_encoder_avx2 PRIVATE
    LXVOXEL_ENCODER_VARIANT=1
    LXVOXEL_ENCODER_ENTRY=lxvoxel_encoder_avx2_main)
  if(LXVOXEL_SINGLE_THREAD)
    target_compile_definitions(lxvoxel_encoder_avx2 PRIVATE LXVOXEL_SINGLE_THREAD=1)
  endif()
  lxvoxel_apply_avx2_optimizations(lxvoxel_encoder_avx2)

  set(LXVOXEL_ENCODER_OBJECTS
    $<TARGET_OBJECTS:lxvoxel_encoder_sse41>
    $<TARGET_OBJECTS:lxvoxel_encoder_avx2>)

  add_library(lxvoxel_decoder_sse41 OBJECT src/decoder/decoder_core.cpp)
  set_target_properties(lxvoxel_decoder_sse41 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(lxvoxel_decoder_sse41 PRIVATE lxvoxel_internal lxvoxel_metadata)
  target_compile_definitions(lxvoxel_decoder_sse41 PRIVATE LXVOXEL_DECODER_ENTRY=decode_stream_sse41)
  lxvoxel_apply_sse41_optimizations(lxvoxel_decoder_sse41)

  add_library(lxvoxel_decoder_avx2 OBJECT src/decoder/decoder_core.cpp)
  set_target_properties(lxvoxel_decoder_avx2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(lxvoxel_decoder_avx2 PRIVATE lxvoxel_internal lxvoxel_metadata)
  target_compile_definitions(lxvoxel_decoder_avx2 PRIVATE LXVOXEL_DECODER_ENTRY=decode_stream_avx2)
  lxvoxel_apply_avx2_optimizations(lxvoxel_decoder_avx2)

  set(LXVOXEL_DECODER_OBJECTS
    $<TARGET_OBJECTS:lxvoxel_decoder_sse41>
    $<TARGET_OBJECTS:lxvoxel_decoder_avx2>)
else()
  add_library(lxvoxel_encoder_neon OBJECT src/encoder/main.cpp)
  target_link_libraries(lxvoxel_encoder_neon PRIVATE lxvoxel_internal)
  target_compile_definitions(lxvoxel_encoder_neon PRIVATE
    LXVOXEL_ENCODER_VARIANT=1
    LXVOXEL_ENCODER_ENTRY=lxvoxel_encoder_neon_main)
  lxvoxel_apply_neon_optimizations(lxvoxel_encoder_neon)
  set(LXVOXEL_ENCODER_OBJECTS $<TARGET_OBJECTS:lxvoxel_encoder_neon>)

  add_library(lxvoxel_decoder_neon OBJECT src/decoder/decoder_core.cpp)
  set_target_properties(lxvoxel_decoder_neon PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(lxvoxel_decoder_neon PRIVATE lxvoxel_internal lxvoxel_metadata)
  target_compile_definitions(lxvoxel_decoder_neon PRIVATE LXVOXEL_DECODER_ENTRY=decode_stream_neon)
  lxvoxel_apply_neon_optimizations(lxvoxel_decoder_neon)
  set(LXVOXEL_DECODER_OBJECTS $<TARGET_OBJECTS:lxvoxel_decoder_neon>)
endif()

add_executable(lxvoxel-enc tools/encoder_cli.cpp ${LXVOXEL_ENCODER_OBJECTS})
target_link_libraries(lxvoxel-enc PRIVATE lxvoxel_internal lxvoxel_metadata lxvoxel_audio_input)
if(WIN32)
  target_link_libraries(lxvoxel-enc PRIVATE shell32)
endif()
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(lxvoxel-enc PRIVATE -municode -static -static-libgcc -static-libstdc++)
endif()
lxvoxel_apply_portable_optimizations(lxvoxel-enc)

add_library(lxvoxel_decoder_core STATIC
  src/decoder/dispatch.cpp
  ${LXVOXEL_DECODER_OBJECTS})
set_target_properties(lxvoxel_decoder_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lxvoxel_decoder_core PUBLIC lxvoxel_internal lxvoxel_metadata)
lxvoxel_apply_portable_optimizations(lxvoxel_decoder_core)

if(LXVOXEL_BUILD_SHARED)
  add_library(lxvoxel SHARED src/api/lxvoxel_c.cpp)
  target_compile_definitions(lxvoxel PRIVATE LXVOXEL_SHARED LXVOXEL_BUILDING_DLL)
  target_link_libraries(lxvoxel PRIVATE lxvoxel_decoder_core)
  target_include_directories(lxvoxel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(lxvoxel PROPERTIES OUTPUT_NAME "lxvoxel" PREFIX "")
  if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(lxvoxel PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
  lxvoxel_apply_optimizations(lxvoxel)
endif()

add_executable(lxvoxel-dec tools/decoder_cli.cpp)
target_link_libraries(lxvoxel-dec PRIVATE lxvoxel_decoder_core)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(lxvoxel-dec PRIVATE -static -static-libgcc -static-libstdc++)
endif()
lxvoxel_apply_optimizations(lxvoxel-dec)

include(CheckIPOSupported)
check_ipo_supported(RESULT LXVOXEL_IPO_SUPPORTED OUTPUT LXVOXEL_IPO_ERROR)
if(LXVOXEL_IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set_property(TARGET lxvoxel-enc lxvoxel-dec lxvoxel_decoder_core lxvoxel_metadata lxvoxel_audio_input PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  if(TARGET lxvoxel)
    set_property(TARGET lxvoxel PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

if(LXVOXEL_BUILD_TESTS)
  enable_testing()
  if(UNIX AND NOT WIN32)
    add_test(NAME lxvoxel-smoke COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.sh")
    set_tests_properties(lxvoxel-smoke PROPERTIES ENVIRONMENT "ENC=$<TARGET_FILE:lxvoxel-enc>;DEC=$<TARGET_FILE:lxvoxel-dec>")
  endif()
  if(WIN32)
    add_executable(lxvoxel-test-property-handler tests/test_property_handler.cpp)
    target_link_libraries(lxvoxel-test-property-handler PRIVATE propsys shlwapi ole32 gdi32)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_link_options(lxvoxel-test-property-handler PRIVATE -municode)
    endif()
  endif()
endif()

# Public headers and command-line applications follow the conventional codec layout.
# The DLL is optional; the stand-alone decoder is always available.
include(GNUInstallDirs)
install(TARGETS lxvoxel-enc lxvoxel-dec
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET lxvoxel)
  install(TARGETS lxvoxel
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/lxvoxel/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lxvoxel
    FILES_MATCHING PATTERN "*.h")
endif()
