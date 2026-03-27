namespace llm {

Response LLMClient::sendRequest(const Request& request) const {
    Response response;
    auto startTime = juce::Time::getMillisecondCounterHiRes();

    auto body = buildRequestBody(request);
    auto url = juce::URL(getEndpointUrl()).withPOSTData(body);
    auto headers = getHeaders();

    // Build header string for JUCE URL API
    juce::String headerString;
    for (auto& key : headers.getAllKeys())
        headerString += key + ": " + headers[key] + "\r\n";

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                       .withExtraHeaders(headerString)
                       .withStatusCode(&statusCode);

    auto stream = url.createInputStream(options);

    response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;

    if (stream == nullptr) {
        response.error = "Failed to connect to " + getEndpointUrl();
        return response;
    }

    auto responseText = stream->readEntireStreamAsString();

    if (statusCode < 200 || statusCode >= 300) {
        response.error = "HTTP " + juce::String(statusCode) + ": " + responseText.substring(0, 500);
        return response;
    }

    response = parseResponseBody(responseText);
    response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;
    return response;
}

}  // namespace llm
