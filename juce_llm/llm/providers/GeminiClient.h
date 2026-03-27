#pragma once

namespace llm {

//==============================================================================
/** Google Gemini native API client — generateContent endpoint. */
class GeminiClient : public LLMClient {
  public:
    using LLMClient::LLMClient;
    juce::String getName() const override {
        return "Gemini";
    }

    juce::String buildRequestBody(const Request& request) const override;
    juce::String getEndpointUrl() const override;
    juce::StringPairArray getHeaders() const override;
    Response parseResponseBody(const juce::String& jsonString) const override;
};

}  // namespace llm
