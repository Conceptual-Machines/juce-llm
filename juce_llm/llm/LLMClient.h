#pragma once

namespace llm
{

//==============================================================================
/** Abstract base class for LLM provider clients.
    Each provider implements buildPayload() and parseResponse().
    The base class handles async HTTP via juce::URL on a background thread.
*/
class LLMClient
{
public:
    explicit LLMClient (const ProviderConfig& config) : config_ (config) {}
    virtual ~LLMClient() = default;

    /** Send a request asynchronously. Callback is invoked on the message thread. */
    void sendRequestAsync (const Request& request, ResponseCallback callback);

    /** Send a request synchronously. Blocks the calling thread. */
    Response sendRequest (const Request& request);

    /** Get the provider name for logging/debugging. */
    virtual juce::String getName() const = 0;

protected:
    /** Build the JSON payload for this provider. */
    virtual juce::var buildPayload (const Request& request) const = 0;

    /** Build the full URL for the API endpoint. */
    virtual juce::URL buildUrl() const = 0;

    /** Build HTTP headers for this provider. */
    virtual juce::StringPairArray buildHeaders() const = 0;

    /** Parse the response JSON into a text string. */
    virtual juce::String parseResponse (const juce::var& json) const = 0;

    ProviderConfig config_;
};

} // namespace llm
