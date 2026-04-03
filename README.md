# juce-llm

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Conceptual-Machines/juce-llm/main.svg)](https://results.pre-commit.ci/latest/github/Conceptual-Machines/juce-llm/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A JUCE module for LLM API integration. Provides a unified interface for text generation and structured output across multiple providers, using JUCE's native HTTP and JSON — zero external dependencies.

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

// Send a request (synchronous — call from any thread)
llm::Request request;
request.systemPrompt = "You are a helpful assistant.";
request.userMessage = "Hello!";

auto response = client->sendRequest (request);

if (response.success)
    DBG (response.text);
else
    DBG ("Error: " + response.error);
```

### Structured output

Use `Schema` to get JSON responses matching a defined structure:

```cpp
llm::Request request;
request.systemPrompt = "Extract chord info from the user's request.";
request.userMessage = "Give me a jazz progression in Bb";
request.schema = llm::Schema::object ({
    { "chords", llm::Schema::array (llm::Schema::string()) },
    { "key",    llm::Schema::string() },
    { "tempo",  llm::Schema::number() }
});

auto response = client->sendRequest (request);
auto json = juce::JSON::parse (response.text);

auto key = json["key"].toString();        // "Bb"
auto chords = json["chords"].getArray();  // ["Bbmaj7", "Eb7", ...]
```

### Data interface

For custom HTTP transport, use the data interface directly:

```cpp
auto client = llm::LLMClientFactory::create (config);

auto body    = client->buildRequestBody (request);   // JSON string
auto url     = client->getEndpointUrl();              // URL string
auto headers = client->getHeaders();                  // StringPairArray

// ... your HTTP transport ...

auto response = client->parseResponseBody (jsonResponseString);
```

### Codex compatibility mode

`juce-llm` can also drive the ChatGPT Codex Responses backend through the `OpenAIResponses` client.

Use `Provider::OpenAIResponses` together with `ProviderConfig::useCodexBackend = true`:

```cpp
llm::ProviderConfig config;
config.provider = llm::Provider::OpenAIResponses;
config.baseUrl = "https://chatgpt.com/backend-api/codex";
config.apiKey = "<oauth access token>";
config.model = "codex/gpt-5-codex";
config.useCodexBackend = true;
config.codexAccountId = "<chatgpt account id>";
```

When Codex mode is enabled, the client automatically:

- appends `/responses` to the configured base URL when needed
- strips the optional `codex/` model prefix before sending the request
- sends Codex-specific headers such as `OpenAI-Beta`, `originator`, and `chatgpt-account-id`
- uses the Codex-compatible `input` message-array payload shape
- parses `response.output_text.delta` streaming events
- omits prompt cache fields that are valid on the official Responses API but rejected by Codex

This mode is intended for ChatGPT-authenticated Codex backends, not the standard OpenAI API.

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
