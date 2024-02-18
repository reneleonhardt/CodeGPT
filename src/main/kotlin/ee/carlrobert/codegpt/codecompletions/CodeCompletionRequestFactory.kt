package ee.carlrobert.codegpt.codecompletions

import ee.carlrobert.codegpt.completions.llama.LlamaModel
import ee.carlrobert.codegpt.settings.service.llama.LlamaSettings
import ee.carlrobert.codegpt.settings.service.llama.LlamaSettingsState
import ee.carlrobert.codegpt.settings.service.openai.OpenAISettings
import ee.carlrobert.llm.client.llama.completion.LlamaCompletionRequest
import ee.carlrobert.llm.client.openai.completion.request.OpenAIChatCompletionRequest
import ee.carlrobert.llm.client.openai.completion.request.OpenAIChatCompletionStandardMessage


object CodeCompletionRequestFactory {
    fun buildOpenAIRequest(details: InfillRequestDetails): OpenAIChatCompletionRequest {
        val prompt = InfillPromptTemplate.OPENAI.buildPrompt(
            details.prefix,
            details.suffix
        )
        val settings = OpenAISettings.getCurrentState()
        return OpenAIChatCompletionRequest.Builder(listOf(OpenAIChatCompletionStandardMessage("user", prompt)))
            .setModel(settings.model)
            .setStream(true)
            .setMaxTokens(OpenAISettings.getCurrentState().codeCompletionMaxTokens)
            .setTemperature(0.4)
//            .setOverriddenPath(if(settings.isUsingCustomPath) settings.path else null)
            .build()
    }

    fun buildLlamaRequest(details: InfillRequestDetails): LlamaCompletionRequest {
        val settings = LlamaSettings.getCurrentState()
        val promptTemplate = getLlamaInfillPromptTemplate(settings)
        val prompt = promptTemplate.buildPrompt(details.prefix, details.suffix)
        return LlamaCompletionRequest.Builder(prompt)
            .setN_predict(settings.codeCompletionMaxTokens)
            .setStream(true)
            .setTemperature(0.4)
            .setStop(promptTemplate.stopTokens)
            .build()
    }

    private fun getLlamaInfillPromptTemplate(settings: LlamaSettingsState): InfillPromptTemplate {
        if (!settings.isRunLocalServer) {
            return settings.remoteModelInfillPromptTemplate
        }
        if (settings.isUseCustomModel) {
            return settings.localModelInfillPromptTemplate
        }
        return LlamaModel.findByHuggingFaceModel(settings.huggingFaceModel).infillPromptTemplate
    }
}
