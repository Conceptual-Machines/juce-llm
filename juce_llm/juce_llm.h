/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

  ID:               juce_llm
  vendor:           Conceptual Machines
  version:          0.1.0
  name:             LLM Client
  description:      Unified LLM API client for JUCE — supports OpenAI, Anthropic,
                    Gemini, OpenRouter, and local llama-server.
  website:          https://github.com/Conceptual-Machines/juce-llm
  license:          MIT
  minimumCppStandard: 20

  dependencies:     juce_core

 END_JUCE_MODULE_DECLARATION
*******************************************************************************/

#pragma once

#include <juce_core/juce_core.h>

#include "llm/LLMClient.h"
#include "llm/LLMClientFactory.h"
#include "llm/LLMTypes.h"
#include "llm/Schema.h"
#include "llm/providers/AnthropicClient.h"
#include "llm/providers/GeminiClient.h"
#include "llm/providers/OpenAIChatClient.h"
#include "llm/providers/OpenAIResponsesClient.h"
