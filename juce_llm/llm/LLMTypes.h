#pragma once

namespace llm {

//==============================================================================
enum class Provider {
    OpenAIChat,       // Chat Completions — DeepSeek, OpenRouter, local llama-server
    OpenAIResponses,  // Responses API — GPT-5+
    Anthropic,        // Messages API — Claude models
    Gemini            // generateContent — Gemini models
};

//==============================================================================
struct ProviderConfig {
    Provider provider;
    juce::String baseUrl;
    juce::String apiKey;
    juce::String model;

    int maxTokens = 4096;  // max output tokens (0 = provider default)

    // Provider-specific options
    bool noTemperature = false;    // GPT-5 doesn't support temperature
    juce::String reasoningEffort;  // "none", "low", "medium", "high", "xhigh"
    juce::String grammar;          // GBNF grammar for llama-server
    int connectionTimeoutMs = 0;   // 0 = system default; useful for local server checks

    // Application identity — used for User-Agent and provider-specific headers
    juce::String userAgent;  // e.g. "MAGDA/0.3.0"
    juce::String appUrl;     // e.g. "https://magda.dev" (for OpenRouter HTTP-Referer)
};

//==============================================================================
struct Request {
    juce::String systemPrompt;
    juce::String userMessage;
    float temperature = 0.1f;

    /** Optional JSON schema for structured output.
        Built via Schema::object(), Schema::array(), etc.
        When set, providers will use their native structured output mechanism. */
    juce::var schema;

    /** Optional CFG grammar for constrained output (Lark format for OpenAI Responses API,
        GBNF for llama-server). When set, the provider will constrain the model output to match
        the grammar. For OpenAI Responses, this uses a custom tool with grammar format. */
    juce::String grammar;

    /** Tool name used for CFG grammar-constrained output (OpenAI Responses API).
        Defaults to "grammar_tool" if not set. */
    juce::String grammarToolName;

    /** Tool description for CFG grammar output. If empty, systemPrompt is used. */
    juce::String grammarToolDescription;

    /** When true, sendStreamingRequest will add provider-specific streaming flags. */
    bool stream = false;
};

//==============================================================================
struct Response {
    juce::String text;
    double wallSeconds = 0.0;
    bool success = false;
    juce::String error;
};

//==============================================================================
using ResponseCallback = std::function<void(Response)>;

/** Called for each token/chunk during streaming. Return false to cancel. */
using StreamCallback = std::function<bool(const juce::String& token)>;

}  // namespace llm
