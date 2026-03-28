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

    // CFG grammar-constrained output via custom tool
    if (request.grammar.isNotEmpty()) {
        auto toolName = request.grammarToolName.isNotEmpty() ? request.grammarToolName
                                                             : juce::String("grammar_tool");
        auto toolDesc = request.grammarToolDescription.isNotEmpty() ? request.grammarToolDescription
                                                                    : request.systemPrompt;

        auto* tool = new juce::DynamicObject();
        tool->setProperty("type", "custom");
        tool->setProperty("name", toolName);
        tool->setProperty("description", toolDesc);

        auto* format = new juce::DynamicObject();
        format->setProperty("type", "grammar");
        format->setProperty("syntax", "lark");
        format->setProperty("definition", request.grammar);
        tool->setProperty("format", juce::var(format));

        juce::Array<juce::var> tools;
        tools.add(juce::var(tool));
        payload->setProperty("tools", tools);
        payload->setProperty("parallel_tool_calls", false);
    }
    // Structured output via JSON schema
    else if (!request.schema.isVoid()) {
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

    if (auto* output = json["output"].getArray()) {
        for (const auto& item : *output) {
            auto type = item["type"].toString();

            // CFG grammar tool response — extract from custom_tool_call
            if (type == "custom_tool_call") {
                auto input = item["input"].toString().trim();
                if (input.isNotEmpty()) {
                    response.text = input;
                    response.success = true;
                    return response;
                }
            }

            // Standard text response — output[].content[].text
            if (type == "message") {
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
