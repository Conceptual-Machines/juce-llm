namespace llm
{

//==============================================================================
Response LLMClient::sendRequest (const Request& request)
{
    Response response;

    auto url = buildUrl().withPOSTData (juce::JSON::toString (buildPayload (request)));
    auto headers = buildHeaders();

    for (int i = 0; i < headers.size(); ++i)
        url = url.withExtraHeader (headers.getAllKeys()[i], headers.getAllValues()[i]);

    auto startTime = juce::Time::getMillisecondCounterHiRes();

    int statusCode = 0;
    auto result = url.readEntireTextStream (false, &statusCode);

    response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;

    if (statusCode >= 200 && statusCode < 300 && result.isNotEmpty())
    {
        auto json = juce::JSON::parse (result);

        if (json.isObject())
        {
            response.text = parseResponse (json);
            response.success = response.text.isNotEmpty();
        }
        else
        {
            response.error = "Failed to parse JSON response";
        }
    }
    else
    {
        response.error = "HTTP " + juce::String (statusCode) + ": " + result.substring (0, 200);
    }

    return response;
}

//==============================================================================
void LLMClient::sendRequestAsync (const Request& request, ResponseCallback callback)
{
    // Capture what we need and run on a background thread
    auto* client = this;
    auto req = request;

    juce::Thread::launch ([client, req, cb = std::move (callback)]()
    {
        auto response = client->sendRequest (req);

        juce::MessageManager::callAsync ([cb, response]()
        {
            cb (response);
        });
    });
}

} // namespace llm
