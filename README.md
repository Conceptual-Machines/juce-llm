# juce-llm

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Conceptual-Machines/juce-llm/main.svg)](https://results.pre-commit.ci/latest/github/Conceptual-Machines/juce-llm/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A JUCE module for LLM API integration. Provides a unified interface for text generation,
structured output, and executable tool calls across multiple providers, using JUCE's native HTTP
and JSON — zero external dependencies.

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

### Executable tools

The same definition works with OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and
Gemini:

```cpp
llm::Request request;
request.userMessage = "What is the weather in London?";
request.tools = {{
    "get_weather",
    "Get the current weather for a city",
    llm::Schema::object ({{ "city", llm::Schema::string() }})
}};

auto response = client->sendRequest (request);
for (const auto& call : response.toolCalls)
{
    if (! call.isValid())
        continue; // call.error contains malformed-argument details

    llm::ToolResult result;
    result.callId = call.id;
    result.name = call.name;
    result.content = getWeather (call.arguments["city"].toString());
    conversation.addToolResult (std::move (result));
}

request.userMessage = {};
response = client->continueConversation (conversation, request);
```

Assistant calls and tool results survive `Conversation::toVar()` / `fromVar()`. For streaming agent
loops, use `sendStreamingRequestDetailed()` to receive `StreamDelta` events; the returned
`Response` contains assembled calls even when argument JSON arrived in fragments or calls were
interleaved.

`Request::grammar` remains a constrained-output mechanism. OpenAI Responses maps it to a custom
grammar tool internally, but it is returned as response text and never as an executable
`ToolCall`.

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
