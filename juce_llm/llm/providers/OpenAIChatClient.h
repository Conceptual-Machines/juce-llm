#pragma once

namespace llm {

//==============================================================================
/** OpenAI Chat Completions client.
    Also works with DeepSeek, OpenRouter, local llama-server,
    and any OpenAI-compatible endpoint.
*/
class OpenAIChatClient : public LLMClient {
  public:
    using LLMClient::LLMClient;
    juce::String getName() const override {
        return "OpenAI Chat";
    }

    juce::String buildRequestBody(const Request& request) const override;
    juce::String getEndpointUrl() const override;
    juce::StringPairArray getHeaders() const override;
    Response parseResponseBody(const juce::String& jsonString) const override;
};

}  // namespace llm
