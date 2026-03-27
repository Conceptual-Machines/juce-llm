#pragma once

namespace llm {

//==============================================================================
/** Abstract base class for LLM provider clients.
    Each provider implements payload building, URL/header generation, and response parsing.
    Includes a synchronous sendRequest() that handles HTTP transport via juce::URL.
*/
class LLMClient {
  public:
    explicit LLMClient(const ProviderConfig& config) : config_(config) {}
    virtual ~LLMClient() = default;

    /** Get the provider name for logging/debugging. */
    virtual juce::String getName() const = 0;

    //==============================================================================
    // Data interface — override these in each provider

    /** Build the JSON payload string for this provider. */
    virtual juce::String buildRequestBody(const Request& request) const = 0;

    /** Get the full endpoint URL. */
    virtual juce::String getEndpointUrl() const = 0;

    /** Get the HTTP headers as key-value pairs. */
    virtual juce::StringPairArray getHeaders() const = 0;

    /** Parse a JSON response string into a Response. */
    virtual Response parseResponseBody(const juce::String& jsonString) const = 0;

    //==============================================================================
    // HTTP transport

    /** Synchronous HTTP request. Call from any thread. */
    Response sendRequest(const Request& request) const;

    /** Access the config. */
    const ProviderConfig& getConfig() const {
        return config_;
    }

  protected:
    ProviderConfig config_;
};

}  // namespace llm
