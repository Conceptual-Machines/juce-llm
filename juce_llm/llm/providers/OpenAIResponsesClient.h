#pragma once

namespace llm
{

//==============================================================================
/** OpenAI Responses API client — for GPT-5 and newer reasoning models. */
class OpenAIResponsesClient : public LLMClient
{
public:
    using LLMClient::LLMClient;
    juce::String getName() const override { return "OpenAI Responses"; }

protected:
    juce::var buildPayload (const Request& request) const override;
    juce::URL buildUrl() const override;
    juce::StringPairArray buildHeaders() const override;
    juce::String parseResponse (const juce::var& json) const override;
};

} // namespace llm
