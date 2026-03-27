#pragma once

namespace llm
{

//==============================================================================
/** Creates LLMClient instances from a ProviderConfig. */
class LLMClientFactory
{
public:
    /** Create a client for the given provider config. */
    static std::unique_ptr<LLMClient> create (const ProviderConfig& config);
};

} // namespace llm
