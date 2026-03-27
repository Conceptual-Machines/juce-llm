#pragma once

namespace llm {

//==============================================================================
/** Anthropic Messages API client — for Claude models. */
class AnthropicClient : public LLMClient {
  public:
    using LLMClient::LLMClient;
    juce::String getName() const override {
        return "Anthropic";
    }

    juce::String buildRequestBody(const Request& request) const override;
    juce::String getEndpointUrl() const override;
    juce::StringPairArray getHeaders() const override;
    Response parseResponseBody(const juce::String& jsonString) const override;
};

}  // namespace llm
