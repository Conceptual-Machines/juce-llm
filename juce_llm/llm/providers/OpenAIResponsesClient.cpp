namespace llm {

juce::String OpenAIResponsesClient::buildRequestBody(const Request& request) const {
    auto* payload = new juce::DynamicObject();
    payload->setProperty("model", config_.model);
    payload->setProperty("instructions", request.systemPrompt);
    payload->setProperty("input", request.userMessage);

    if (!config_.noTemperature)
        payload->setProperty("temperature", (double)request.temperature);

    if (config_.reasoningEffort.isNotEmpty()) {
        auto* reasoning = new juce::DynamicObject();
        reasoning->setProperty("effort", config_.reasoningEffort);
        payload->setProperty("reasoning", juce::var(reasoning));
    }

    // Structured output via JSON schema
    if (!request.schema.isVoid()) {
        auto* schemaWrapper = new juce::DynamicObject();
        schemaWrapper->setProperty("name", "response");
        schemaWrapper->setProperty("strict", true);
        schemaWrapper->setProperty("schema", request.schema);

        auto* text = new juce::DynamicObject();
        text->setProperty("type", "json_schema");
        text->setProperty("json_schema", juce::var(schemaWrapper));

        payload->setProperty("text", juce::var(text));
    }

    return juce::JSON::toString(juce::var(payload), true);
}

juce::String OpenAIResponsesClient::getEndpointUrl() const {
    return config_.baseUrl + "/responses";
}

juce::StringPairArray OpenAIResponsesClient::getHeaders() const {
    juce::StringPairArray headers;
    headers.set("Authorization", "Bearer " + config_.apiKey);
    headers.set("Content-Type", "application/json");
    return headers;
}

Response OpenAIResponsesClient::parseResponseBody(const juce::String& jsonString) const {
    Response response;
    auto json = juce::JSON::parse(jsonString);

    // Responses API nests text inside output[].content[].text
    if (auto* output = json["output"].getArray()) {
        for (const auto& item : *output) {
            if (item["type"].toString() == "message") {
                if (auto* content = item["content"].getArray()) {
                    for (const auto& c : *content) {
                        if (c["type"].toString() == "output_text") {
                            response.text = c["text"].toString().trim();
                            response.success = response.text.isNotEmpty();
                            return response;
                        }
                    }
                }
            }
        }
    }

    response.error = "Failed to parse response: " + jsonString.substring(0, 200);
    return response;
}

}  // namespace llm
