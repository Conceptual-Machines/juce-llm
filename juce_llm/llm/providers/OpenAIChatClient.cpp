namespace llm {

juce::String OpenAIChatClient::buildRequestBody(const Request& request) const {
    auto messagesArray = juce::Array<juce::var>();

    auto* sysMsg = new juce::DynamicObject();
    sysMsg->setProperty("role", "system");
    sysMsg->setProperty("content", request.systemPrompt);
    messagesArray.add(juce::var(sysMsg));

    auto* userMsg = new juce::DynamicObject();
    userMsg->setProperty("role", "user");
    userMsg->setProperty("content", request.userMessage);
    messagesArray.add(juce::var(userMsg));

    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    payload->setProperty("messages", messagesArray);
    payload->setProperty("temperature", (double)request.temperature);

    // GBNF grammar for llama-server (per-config or per-request)
    auto grammar = config_.grammar.isNotEmpty() ? config_.grammar : request.grammar;
    if (grammar.isNotEmpty())
        payload->setProperty("grammar", grammar);

    // Structured output via JSON schema
    if (!request.schema.isVoid()) {
        auto* schemaWrapper = new juce::DynamicObject();
        schemaWrapper->setProperty("name", "response");
        schemaWrapper->setProperty("strict", true);
        schemaWrapper->setProperty("schema", request.schema);

        auto* responseFormat = new juce::DynamicObject();
        responseFormat->setProperty("type", "json_schema");
        responseFormat->setProperty("json_schema", juce::var(schemaWrapper));

        payload->setProperty("response_format", juce::var(responseFormat));
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String OpenAIChatClient::getEndpointUrl() const {
    return config_.baseUrl + "/chat/completions";
}

juce::StringPairArray OpenAIChatClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Authorization", "Bearer " + config_.apiKey);
    headers.set("Content-Type", "application/json");
    return headers;
}

Response OpenAIChatClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    if (auto* choices = json["choices"].getArray()) {
        if (choices->size() > 0) {
            response.text = (*choices)[0]["message"]["content"].toString().trim();
            response.success = response.text.isNotEmpty();
        }
    }

    if (!response.success)
        response.error = "Failed to parse response: " + jsonString.substring(0, 200);

    return response;
}

}  // namespace llm
