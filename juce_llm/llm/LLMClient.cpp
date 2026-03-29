namespace llm {

Response LLMClient::sendRequest(const Request& request) const {
    Response response;
    auto startTime = juce::Time::getMillisecondCounterHiRes();

    auto body = buildRequestBody(request);
    auto url = juce::URL(getEndpointUrl()).withPOSTData(body);
    auto headers = getHeaders();

    // Inject User-Agent if configured
    if (config_.userAgent.isNotEmpty() && !headers.containsKey("User-Agent"))
        headers.set("User-Agent", config_.userAgent);

    // Build header string for JUCE URL API
    juce::String headerString;
    for (auto& key : headers.getAllKeys())
        headerString += key + ": " + headers[key] + "\r\n";

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                       .withExtraHeaders(headerString)
                       .withStatusCode(&statusCode)
                       .withConnectionTimeoutMs(config_.connectionTimeoutMs > 0
                                                    ? config_.connectionTimeoutMs
                                                    : 0);

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

// Default streaming endpoint — same as non-streaming
juce::String LLMClient::getStreamingEndpointUrl() const {
    return getEndpointUrl();
}

// Default streaming body: inject "stream":true into the normal request body JSON
juce::String LLMClient::buildStreamingRequestBody(const Request& request) const {
    auto body = buildRequestBody(request);
    // Insert "stream":true before the closing brace
    auto lastBrace = body.lastIndexOfChar('}');
    if (lastBrace >= 0)
        return body.substring(0, lastBrace) + ",\n\"stream\": true\n}";
    return body;
}

// Default SSE chunk parser for OpenAI-compatible format:
//   data: {"choices":[{"delta":{"content":"token"}}]}
juce::String LLMClient::parseStreamChunk(const juce::String& dataLine) const {
    auto json = juce::JSON::parse(dataLine);
    if (auto* choices = json["choices"].getArray()) {
        if (choices->size() > 0)
            return (*choices)[0]["delta"]["content"].toString();
    }
    return {};
}

Response LLMClient::sendStreamingRequest(const Request& request, StreamCallback onToken) const {
    Response response;
    auto startTime = juce::Time::getMillisecondCounterHiRes();

    auto body = buildStreamingRequestBody(request);
    auto url = juce::URL(getStreamingEndpointUrl()).withPOSTData(body);
    auto headers = getHeaders();

    // Inject User-Agent if configured
    if (config_.userAgent.isNotEmpty() && !headers.containsKey("User-Agent"))
        headers.set("User-Agent", config_.userAgent);

    juce::String headerString;
    for (auto& key : headers.getAllKeys())
        headerString += key + ": " + headers[key] + "\r\n";

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                       .withExtraHeaders(headerString)
                       .withStatusCode(&statusCode)
                       .withConnectionTimeoutMs(config_.connectionTimeoutMs > 0
                                                    ? config_.connectionTimeoutMs
                                                    : 0);

    auto stream = url.createInputStream(options);

    if (stream == nullptr) {
        response.error = "Failed to connect to " + getEndpointUrl();
        response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;
        return response;
    }

    if (statusCode < 200 || statusCode >= 300) {
        auto errText = stream->readEntireStreamAsString();
        response.error = "HTTP " + juce::String(statusCode) + ": " + errText.substring(0, 500);
        response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;
        return response;
    }

    // Read SSE stream line by line
    juce::String accumulated;
    bool cancelled = false;

    // Read into a byte buffer, convert complete lines as UTF-8
    std::vector<char> rawLineBuffer;

    while (!stream->isExhausted() && !cancelled) {
        char c;
        if (stream->read(&c, 1) != 1)
            break;

        if (c == '\n') {
            auto line = juce::String::fromUTF8(rawLineBuffer.data(),
                                                static_cast<int>(rawLineBuffer.size()))
                            .trim();
            rawLineBuffer.clear();

            if (line.startsWith("data: ")) {
                auto data = line.substring(6).trim();
                if (data == "[DONE]")
                    break;

                auto token = parseStreamChunk(data);
                if (token.isNotEmpty()) {
                    accumulated += token;
                    if (onToken && !onToken(token))
                        cancelled = true;
                }
            }
        } else {
            rawLineBuffer.push_back(c);
        }
    }

    response.text = accumulated.trim();
    response.success = response.text.isNotEmpty();
    response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;

    if (cancelled) {
        response.error = "Cancelled";
        response.success = false;
    }

    return response;
}

}  // namespace llm
