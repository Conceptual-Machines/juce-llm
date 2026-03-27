#pragma once

namespace llm
{

//==============================================================================
/** Anthropic Messages API client — for Claude models. */
class AnthropicClient : public LLMClient
{
public:
    using LLMClient::LLMClient;
    juce::String getName() const override { return "Anthropic"; }

protected:
    juce::var buildPayload (const Request& request) const override;
    juce::URL buildUrl() const override;
    juce::StringPairArray buildHeaders() const override;
    juce::String parseResponse (const juce::var& json) const override;
};

} // namespace llm
