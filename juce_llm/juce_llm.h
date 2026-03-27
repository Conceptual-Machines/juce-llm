/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

  ID:               juce_llm
  vendor:           Conceptual Machines
  version:          0.1.0
  name:             LLM Client
  description:      Unified LLM API client for JUCE — text generation and structured
                    output across OpenAI, Anthropic, Gemini, OpenRouter, and llama-server.
  website:          https://github.com/Conceptual-Machines/juce-llm
  license:          MIT
  minimumCppStandard: 20

  dependencies:     juce_core

 END_JUCE_MODULE_DECLARATION
*******************************************************************************/

#pragma once

#include <juce_core/juce_core.h>

// clang-format off
// Include order matters — types before client before providers
#include "llm/LLMTypes.h"
#include "llm/Schema.h"
#include "llm/LLMClient.h"
#include "llm/LLMClientFactory.h"
#include "llm/providers/OpenAIChatClient.h"
#include "llm/providers/OpenAIResponsesClient.h"
#include "llm/providers/AnthropicClient.h"
#include "llm/providers/GeminiClient.h"
// clang-format on
