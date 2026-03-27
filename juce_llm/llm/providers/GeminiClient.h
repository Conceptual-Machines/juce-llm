#pragma once

namespace llm
{

//==============================================================================
/** Google Gemini native API client — generateContent endpoint. */
class GeminiClient : public LLMClient
{
public:
    using LLMClient::LLMClient;
    juce::String getName() const override { return "Gemini"; }

protected:
    juce::var buildPayload (const Request& request) const override;
    juce::URL buildUrl() const override;
    juce::StringPairArray buildHeaders() const override;
    juce::String parseResponse (const juce::var& json) const override;
};

} // namespace llm
