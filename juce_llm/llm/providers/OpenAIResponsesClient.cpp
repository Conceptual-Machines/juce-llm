namespace llm
{

juce::var OpenAIResponsesClient::buildPayload (const Request& request) const
{
    auto* payload = new juce::DynamicObject();
    payload->setProperty ("model", config_.model);
    payload->setProperty ("instructions", request.systemPrompt);
    payload->setProperty ("input", request.userMessage);

    if (! config_.noTemperature)
        payload->setProperty ("temperature", (double) request.temperature);

    if (config_.reasoningEffort.isNotEmpty())
    {
        auto* reasoning = new juce::DynamicObject();
        reasoning->setProperty ("effort", config_.reasoningEffort);
        payload->setProperty ("reasoning", juce::var (reasoning));
    }

    return juce::var (payload);
}

juce::URL OpenAIResponsesClient::buildUrl() const
{
    return juce::URL (config_.baseUrl + "/responses");
}

juce::StringPairArray OpenAIResponsesClient::buildHeaders() const
{
    juce::StringPairArray headers;
    headers.set ("Authorization", "Bearer " + config_.apiKey);
    headers.set ("Content-Type", "application/json");
    return headers;
}

juce::String OpenAIResponsesClient::parseResponse (const juce::var& json) const
{
    // Responses API nests text inside output[].content[].text
    if (auto* output = json["output"].getArray())
    {
        for (const auto& item : *output)
        {
            if (item["type"].toString() == "message")
            {
                if (auto* content = item["content"].getArray())
                {
                    for (const auto& c : *content)
                    {
                        if (c["type"].toString() == "output_text")
                            return c["text"].toString().trim();
                    }
                }
            }
        }
    }

    return {};
}

} // namespace llm
