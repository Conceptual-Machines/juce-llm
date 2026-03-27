# juce-llm

A JUCE module for LLM API integration. Provides a unified interface for text generation across multiple providers, using JUCE's native HTTP and JSON — zero external dependencies.

## Providers

| Implementation | Providers | Notes |
|---|---|---|
| **OpenAI Chat Completions** | OpenAI, DeepSeek, OpenRouter, local llama-server, Ollama, LM Studio | Any OpenAI-compatible endpoint |
| **OpenAI Responses API** | GPT-5+ | Reasoning models, `reasoning.effort` support |
| **Anthropic Messages** | Claude models | |
| **Gemini native** | Gemini models | `generateContent` endpoint |

## Usage

```cpp
#include <juce_llm/juce_llm.h>

// Create a client
llm::ProviderConfig config;
config.provider = llm::Provider::OpenAIChat;
config.baseUrl = "https://openrouter.ai/api/v1";
config.apiKey = "sk-or-...";
config.model = "meta-llama/llama-3.3-70b-instruct";

auto client = llm::LLMClientFactory::create (config);

// Send a request (async)
llm::Request request;
request.systemPrompt = "You are a helpful assistant.";
request.userMessage = "Hello!";

client->sendRequestAsync (request, [] (llm::Response response)
{
    if (response.success)
        DBG (response.text);
    else
        DBG ("Error: " + response.error);
});

// Or synchronously (on a background thread)
auto response = client->sendRequest (request);
```

## Local llama-server

```cpp
llm::ProviderConfig config;
config.provider = llm::Provider::OpenAIChat;
config.baseUrl = "http://127.0.0.1:8080/v1";
config.model = "local";
config.grammar = myGBNFGrammar;  // Optional GBNF constraint

auto client = llm::LLMClientFactory::create (config);
```

## Adding to your project

### CMake (recommended)

```cmake
add_subdirectory(path/to/juce-llm)
target_link_libraries(MyTarget PRIVATE juce_llm)
```

### As a JUCE module

Add the `juce_llm` folder to your module search paths. The module depends only on `juce_core`.

## Requirements

- C++20
- JUCE 7+

## License

MIT
