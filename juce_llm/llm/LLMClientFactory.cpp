namespace llm {

std::unique_ptr<LLMClient> LLMClientFactory::create(const ProviderConfig& config) {
    switch (config.provider) {
        case Provider::OpenAIChat:
            return std::make_unique<OpenAIChatClient>(config);

        case Provider::OpenAIResponses:
            return std::make_unique<OpenAIResponsesClient>(config);

        case Provider::Anthropic:
            return std::make_unique<AnthropicClient>(config);

        case Provider::Gemini:
            return std::make_unique<GeminiClient>(config);
    }

    jassertfalse;
    return nullptr;
}

}  // namespace llm
