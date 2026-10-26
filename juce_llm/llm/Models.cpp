namespace llm {

namespace {
// Temperature: the Claude 5 generation (and Opus 4.7+) reject it, as do every
// Responses-API OpenAI model; Gemini and DeepSeek accept it.
const std::vector<ModelInfo> kCatalog = {
    // Anthropic
    {"claude-fable-5-1", "Claude Fable 5.1", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Frontier, false, false},
    {"claude-opus-5-5", "Claude Opus 5.5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Flagship, false, false},
    {"claude-sonnet-5-5", "Claude Sonnet 5.5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Balanced, false, false},
    {"claude-haiku-5-5", "Claude Haiku 5.5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Fast, false, false},
    {"claude-fable-5", "Claude Fable 5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Frontier, false, true},
    {"claude-opus-5", "Claude Opus 5", Vendor::Anthropic, Provider::Anthropic, ModelTier::Flagship,
     false, true},
    {"claude-opus-4-8", "Claude Opus 4.8", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Flagship, false, true},
    {"claude-sonnet-5", "Claude Sonnet 5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Balanced, false, true},
    {"claude-sonnet-4-6", "Claude Sonnet 4.6", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Balanced, true, true},
    {"claude-haiku-4-5-20251001", "Claude Haiku 4.5", Vendor::Anthropic, Provider::Anthropic,
     ModelTier::Fast, true, true},

    // OpenAI (Responses API only)
    {"gpt-6-astra", "GPT-6 Astra", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Frontier,
     false, false},
    {"gpt-6.1-sol", "GPT-6.1 Sol", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Flagship,
     false, false},
    {"gpt-6.1-terra", "GPT-6.1 Terra", Vendor::OpenAI, Provider::OpenAIResponses,
     ModelTier::Balanced, false, false},
    {"gpt-6-luna", "GPT-6 Luna", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Fast, false,
     false},
    {"gpt-5.6-sol", "GPT-5.6 Sol", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Flagship,
     false, true},
    {"gpt-5.6-terra", "GPT-5.6 Terra", Vendor::OpenAI, Provider::OpenAIResponses,
     ModelTier::Balanced, false, true},
    {"gpt-5.6-luna", "GPT-5.6 Luna", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Fast,
     false, true},
    {"gpt-5.5", "GPT-5.5", Vendor::OpenAI, Provider::OpenAIResponses, ModelTier::Flagship, false,
     true},

    // Google
    {"gemini-4-argon", "Gemini 4 Argon", Vendor::Google, Provider::Gemini, ModelTier::Flagship,
     true, false},
    {"gemini-3.8-flash", "Gemini 3.8 Flash", Vendor::Google, Provider::Gemini, ModelTier::Balanced,
     true, false},
    {"gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", Vendor::Google, Provider::Gemini,
     ModelTier::Fast, true, false},
    {"gemini-3.6-flash", "Gemini 3.6 Flash", Vendor::Google, Provider::Gemini, ModelTier::Balanced,
     true, true},
    {"gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", Vendor::Google, Provider::Gemini,
     ModelTier::Fast, true, true},
    {"gemini-2.5-pro", "Gemini 2.5 Pro", Vendor::Google, Provider::Gemini, ModelTier::Flagship,
     true, true},
    {"gemini-2.5-flash", "Gemini 2.5 Flash", Vendor::Google, Provider::Gemini, ModelTier::Balanced,
     true, true},

    // DeepSeek (OpenAI Chat Completions)
    {"deepseek-v4-pro", "DeepSeek V4 Pro", Vendor::DeepSeek, Provider::OpenAIChat,
     ModelTier::Flagship, true, false},
    {"deepseek-v4-flash", "DeepSeek V4 Flash", Vendor::DeepSeek, Provider::OpenAIChat,
     ModelTier::Fast, true, false},
};
}  // namespace

const std::vector<ModelInfo>& modelCatalog() {
    return kCatalog;
}

const ModelInfo* findModel(const juce::String& id) {
    for (const auto& m : kCatalog)
        if (id == m.id)
            return &m;
    return nullptr;
}

std::vector<ModelInfo> modelsFor(Vendor vendor, bool includeLegacy) {
    std::vector<ModelInfo> out;
    for (const auto& m : kCatalog)
        if (m.vendor == vendor && (includeLegacy || !m.legacy))
            out.push_back(m);
    return out;
}

const ModelInfo& defaultModel(Vendor vendor, ModelTier tier) {
    const ModelInfo* best = nullptr;
    int bestDistance = 0;
    for (const auto& m : kCatalog) {
        if (m.vendor != vendor || m.legacy)
            continue;
        const int d = std::abs(static_cast<int>(m.tier) - static_cast<int>(tier));
        if (best == nullptr || d < bestDistance) {
            best = &m;
            bestDistance = d;
        }
    }
    jassert(best != nullptr);  // every vendor has a current model
    return best != nullptr ? *best : kCatalog.front();
}

juce::String defaultBaseUrl(Vendor vendor) {
    switch (vendor) {
        case Vendor::Anthropic:
            return "https://api.anthropic.com/v1";
        case Vendor::Google:
            return "https://generativelanguage.googleapis.com";
        case Vendor::DeepSeek:
            return "https://api.deepseek.com";
        case Vendor::OpenAI:
            break;
    }
    return "https://api.openai.com/v1";
}

juce::String vendorName(Vendor vendor) {
    switch (vendor) {
        case Vendor::Anthropic:
            return "Anthropic";
        case Vendor::Google:
            return "Google";
        case Vendor::DeepSeek:
            return "DeepSeek";
        case Vendor::OpenAI:
            break;
    }
    return "OpenAI";
}

bool requiresOpenAIResponsesAPI(const juce::String& modelId) {
    if (modelId.startsWith("gpt-5") || modelId.startsWith("gpt-6"))
        return true;
    return modelId.length() > 1 && modelId[0] == 'o' &&
           juce::CharacterFunctions::isDigit(modelId[1]);
}

ProviderConfig makeConfig(const ModelInfo& model, const juce::String& apiKey) {
    ProviderConfig c;
    c.provider = model.provider;
    c.baseUrl = defaultBaseUrl(model.vendor);
    c.apiKey = apiKey;
    c.model = model.id;
    c.noTemperature = !model.acceptsTemperature;
    return c;
}

}  // namespace llm
