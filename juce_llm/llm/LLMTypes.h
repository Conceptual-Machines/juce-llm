#pragma once

namespace llm
{

//==============================================================================
enum class Provider
{
    OpenAIChat,       // Chat Completions — DeepSeek, OpenRouter, local llama-server
    OpenAIResponses,  // Responses API — GPT-5+
    Anthropic,        // Messages API — Claude models
    Gemini            // generateContent — Gemini models
};

//==============================================================================
struct ProviderConfig
{
    Provider provider;
    juce::String baseUrl;
    juce::String apiKey;
    juce::String model;

    // Provider-specific options
    bool noTemperature = false;           // GPT-5 doesn't support temperature
    juce::String reasoningEffort;         // "minimal", "low", "medium", "high"
    juce::String grammar;                 // GBNF grammar for llama-server
};

//==============================================================================
struct Request
{
    juce::String systemPrompt;
    juce::String userMessage;
    float temperature = 0.1f;
};

//==============================================================================
struct Response
{
    juce::String text;
    double wallSeconds = 0.0;
    bool success = false;
    juce::String error;
};

//==============================================================================
using ResponseCallback = std::function<void (Response)>;

} // namespace llm
