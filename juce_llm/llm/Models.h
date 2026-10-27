#pragma once

#include <vector>

namespace llm {

//==============================================================================
/** Who serves a model. Vendors can share one wire API (DeepSeek speaks OpenAI
    Chat Completions), so the vendor picks the base URL and the key, the
    Provider picks the payload. OpenRouter and local servers take free-text
    ids, so they have no catalog entries - build their ProviderConfig by hand. */
enum class Vendor { Anthropic, OpenAI, Google, DeepSeek };

/** A model's place in its vendor's lineup - for a picker's ordering and for a
    caller that asks for "the fast one" without naming an id. */
enum class ModelTier { Frontier, Flagship, Balanced, Fast };

//==============================================================================
/** One hosted model the library knows how to call. `id` is what goes in the
    request; `legacy` marks a model that still answers but has a newer
    replacement in the same line. */
struct ModelInfo {
    const char* id;
    const char* label;
    Vendor vendor;
    Provider provider;
    ModelTier tier;
    bool acceptsTemperature;
    bool legacy;
};

/** Every model in the catalog, current models first within each vendor. */
const std::vector<ModelInfo>& modelCatalog();

/** The catalog entry for `id`, or nullptr when the id is not in it (a custom
    or OpenRouter id - still callable through a hand-built ProviderConfig). */
const ModelInfo* findModel(const juce::String& id);

/** The vendor's models, current first. `includeLegacy` false drops the
    superseded ones. */
std::vector<ModelInfo> modelsFor(Vendor vendor, bool includeLegacy = false);

/** The vendor's recommended model at `tier` (the closest tier it has when it
    has none at exactly that one). */
const ModelInfo& defaultModel(Vendor vendor, ModelTier tier = ModelTier::Balanced);

/** The vendor's public API root, as ProviderConfig::baseUrl expects it. */
juce::String defaultBaseUrl(Vendor vendor);

juce::String vendorName(Vendor vendor);

/** OpenAI models that only the Responses API serves (gpt-5 onward and the
    o-series reasoning models). */
bool requiresOpenAIResponsesAPI(const juce::String& modelId);

/** A ready ProviderConfig for a catalog model: provider, base URL, model id
    and the temperature gate filled from the entry. */
ProviderConfig makeConfig(const ModelInfo& model, const juce::String& apiKey);

}  // namespace llm
