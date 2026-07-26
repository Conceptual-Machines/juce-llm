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
                       .withConnectionTimeoutMs(
                           config_.connectionTimeoutMs > 0 ? config_.connectionTimeoutMs : 0);

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

std::vector<StreamDelta> LLMClient::parseStreamDeltas(const juce::String& dataLine) const {
    auto token = parseStreamChunk(dataLine);
    if (token.isEmpty())
        return {};
    StreamDelta delta;
    delta.type = StreamDeltaType::Text;
    delta.text = token;
    return {delta};
}

Response LLMClient::sendStreamingRequest(const Request& request, StreamCallback onToken) const {
    return sendStreamingRequestDetailed(request,
                                        [callback = std::move(onToken)](const StreamDelta& delta) {
                                            if (delta.type != StreamDeltaType::Text || !callback)
                                                return true;
                                            return callback(delta.text);
                                        });
}

Response LLMClient::sendStreamingRequestDetailed(const Request& request,
                                                 StreamDeltaCallback onDelta) const {
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
                       .withConnectionTimeoutMs(
                           config_.connectionTimeoutMs > 0 ? config_.connectionTimeoutMs : 0);

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
    StreamAccumulator accumulator;
    bool cancelled = false;

    // Read into a byte buffer, convert complete lines as UTF-8
    std::vector<char> rawLineBuffer;

    while (!stream->isExhausted() && !cancelled) {
        char c;
        if (stream->read(&c, 1) != 1)
            break;

        if (c == '\n') {
            auto line =
                juce::String::fromUTF8(rawLineBuffer.data(), static_cast<int>(rawLineBuffer.size()))
                    .trim();
            rawLineBuffer.clear();

            if (line.startsWith("data: ")) {
                auto data = line.substring(6).trim();
                if (data == "[DONE]")
                    break;

                for (const auto& delta : parseStreamDeltas(data)) {
                    accumulator.add(delta);
                    if (onDelta && !onDelta(delta)) {
                        cancelled = true;
                        break;
                    }
                }
            }
        } else {
            rawLineBuffer.push_back(c);
        }
    }

    response = accumulator.finish();
    response.wallSeconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;

    if (cancelled) {
        response.error = "Cancelled";
        response.success = false;
    }

    return response;
}

//==============================================================================
namespace {
// Feed the conversation state into the outgoing request. Both fields are set;
// each provider's buildRequestBody uses only the one its API supports
// (messages[] for stateless providers, previousResponseId for Responses).
void applyConversation(const Conversation& conv, Request& request) {
    request.messages = conv.messages;
    request.previousResponseId = conv.lastResponseId;
}

// Record a completed turn back into the conversation: the user prompt and the
// assistant reply (kept for stateless replay + display), plus the response id
// (used by the Responses API to chain the next turn).
void recordTurn(Conversation& conv, const Request& request, const Response& response) {
    if (request.userMessage.isNotEmpty())
        conv.messages.push_back({"user", request.userMessage});
    Message assistant;
    assistant.role = "assistant";
    assistant.content = response.text;
    assistant.toolCalls = response.toolCalls;
    conv.messages.push_back(std::move(assistant));
    if (response.id.isNotEmpty())
        conv.lastResponseId = response.id;
}
}  // namespace

Response LLMClient::continueConversation(Conversation& conv, Request request) const {
    applyConversation(conv, request);
    auto response = sendRequest(request);
    if (response.success)
        recordTurn(conv, request, response);
    return response;
}

Response LLMClient::continueConversationStreaming(Conversation& conv, Request request,
                                                  StreamCallback onToken) const {
    applyConversation(conv, request);
    auto response = sendStreamingRequest(request, std::move(onToken));
    if (response.success)
        recordTurn(conv, request, response);
    return response;
}

Response LLMClient::continueConversationStreamingDetailed(Conversation& conv, Request request,
                                                          StreamDeltaCallback onDelta) const {
    applyConversation(conv, request);
    auto response = sendStreamingRequestDetailed(request, std::move(onDelta));
    if (response.success)
        recordTurn(conv, request, response);
    return response;
}

}  // namespace llm
