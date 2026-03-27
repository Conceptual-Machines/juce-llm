#pragma once

namespace llm {

//==============================================================================
/** Helper to build JSON Schema objects as juce::var for structured LLM output.

    Usage:
    @code
    auto schema = Schema::object ({
        { "chords", Schema::array (Schema::string()) },
        { "key",    Schema::string() },
        { "tempo",  Schema::number() }
    });
    @endcode
*/
class Schema {
  public:
    static juce::var string() {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "string");
        return juce::var(obj);
    }

    static juce::var number() {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "number");
        return juce::var(obj);
    }

    static juce::var integer() {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "integer");
        return juce::var(obj);
    }

    static juce::var boolean() {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "boolean");
        return juce::var(obj);
    }

    static juce::var array(const juce::var& items) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "array");
        obj->setProperty("items", items);
        return juce::var(obj);
    }

    /** Build an object schema from name-schema pairs. */
    static juce::var object(std::initializer_list<std::pair<juce::String, juce::var>> fields) {
        auto* properties = new juce::DynamicObject();
        auto required = juce::Array<juce::var>();

        for (const auto& [name, fieldSchema] : fields) {
            properties->setProperty(name, fieldSchema);
            required.add(name);
        }

        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "object");
        obj->setProperty("properties", juce::var(properties));
        obj->setProperty("required", required);
        obj->setProperty("additionalProperties", false);
        return juce::var(obj);
    }

    /** Build an enum schema (string with allowed values). */
    static juce::var oneOf(std::initializer_list<juce::String> values) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("type", "string");

        auto enumArray = juce::Array<juce::var>();
        for (const auto& v : values)
            enumArray.add(v);

        obj->setProperty("enum", enumArray);
        return juce::var(obj);
    }

    /** Convert a schema var to its JSON string representation. */
    static juce::String toJsonString(const juce::var& schema) {
        return juce::JSON::toString(schema, true);
    }
};

}  // namespace llm
