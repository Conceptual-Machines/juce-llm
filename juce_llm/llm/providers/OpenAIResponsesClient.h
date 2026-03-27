#pragma once

namespace llm {

//==============================================================================
/** OpenAI Responses API client — for GPT-5 and newer reasoning models. */
class OpenAIResponsesClient : public LLMClient {
  public:
    using LLMClient::LLMClient;
    juce::String getName() const override {
        return "OpenAI Responses";
    }

    juce::String buildRequestBody(const Request& request) const override;
    juce::String getEndpointUrl() const override;
    juce::StringPairArray getHeaders() const override;
    Response parseResponseBody(const juce::String& jsonString) const override;
};

}  // namespace llm
