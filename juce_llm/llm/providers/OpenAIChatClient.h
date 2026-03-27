#pragma once

namespace llm
{

//==============================================================================
/** OpenAI Chat Completions client.
    Also works with DeepSeek, OpenRouter, local llama-server,
    and any OpenAI-compatible endpoint.
*/
class OpenAIChatClient : public LLMClient
{
public:
    using LLMClient::LLMClient;
    juce::String getName() const override { return "OpenAI Chat"; }

protected:
    juce::var buildPayload (const Request& request) const override;
    juce::URL buildUrl() const override;
    juce::StringPairArray buildHeaders() const override;
    juce::String parseResponse (const juce::var& json) const override;
};

} // namespace llm
