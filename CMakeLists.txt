cmake_minimum_required(VERSION 3.22)
project(juce_llm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build the test app
option(JUCE_LLM_BUILD_TESTS "Build test console app" OFF)

# Find JUCE — user must provide JUCE path
if(NOT DEFINED JUCE_PATH)
    # Try common locations
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE")
        set(JUCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JUCE")
        set(JUCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/JUCE")
    else()
        message(FATAL_ERROR "JUCE not found. Set JUCE_PATH to your JUCE directory.")
    endif()
endif()

add_subdirectory(${JUCE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/JUCE)

# Define the JUCE module
juce_add_module(${CMAKE_CURRENT_SOURCE_DIR}/juce_llm)

if(JUCE_LLM_BUILD_TESTS)
    juce_add_console_app(juce_llm_test PRODUCT_NAME "juce_llm_test")

    target_sources(juce_llm_test PRIVATE tests/main.cpp)

    target_compile_definitions(juce_llm_test PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    )

    target_link_libraries(juce_llm_test PRIVATE
        juce_llm
        juce::juce_core
        juce::juce_recommended_config_flags
    )
endif()
