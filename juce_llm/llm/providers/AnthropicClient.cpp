namespace llm {

juce::String AnthropicClient::buildRequestBody(const Request& request) const {
    auto messagesArray = juce::Array<juce::var>();

    auto* userMsg = new juce::DynamicObject();
    userMsg->setProperty("role", "user");
    userMsg->setProperty("content", request.userMessage);
    messagesArray.add(juce::var(userMsg));

    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    payload->setProperty("max_tokens", 1024);
    payload->setProperty("temperature", (double)request.temperature);
    payload->setProperty("system", request.systemPrompt);
    payload->setProperty("messages", messagesArray);

    // Anthropic doesn't have native JSON schema mode — inject schema into system prompt
    // The caller can also use tool_use for structured output, but that's more advanced

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String AnthropicClient::getEndpointUrl() const {
    return config_.baseUrl + "/messages";
}

juce::StringPairArray AnthropicClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("x-api-key", config_.apiKey);
    headers.set("anthropic-version", "2023-06-01");
    headers.set("Content-Type", "application/json");
    return headers;
}

Response AnthropicClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* content = json["content"].getArray()) {
        if (content->size() > 0) {
            response.text = (*content)[0]["text"].toString().trim();
            response.success = response.text.isNotEmpty();
        }
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

}  // namespace llm
