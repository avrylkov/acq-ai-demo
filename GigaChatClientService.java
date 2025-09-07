package org.example.agent;

import chat.giga.client.GigaChatClient;
import chat.giga.client.auth.AuthClient;
import chat.giga.client.auth.AuthClientBuilder;
import chat.giga.model.ModelName;
import chat.giga.model.completion.ChatMessage;
import chat.giga.model.completion.ChatMessageRole;
import chat.giga.model.completion.CompletionRequest;
import chat.giga.model.completion.CompletionResponse;
import chat.giga.model.embedding.EmbeddingRequest;
import chat.giga.model.embedding.EmbeddingResponse;
import dev.langchain4j.data.embedding.Embedding;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

import static chat.giga.model.Scope.GIGACHAT_API_PERS;

@Service
public class GigaChatClientService {

    @Value("${giga.auth-key}")
    private String key;
    private GigaChatClient gigachatClient;

    @PostConstruct
    private void init() {
        gigachatClient = GigaChatClient.builder()
                .authClient(AuthClient.builder()
                        .withOAuth(AuthClientBuilder.OAuthBuilder.builder()
                                .authApiUrl("https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2")
                                .authKey(key)
                                .scope(GIGACHAT_API_PERS)
                                .build())
                        .build())
                .logRequests(true)
                .logResponses(true)
                .build();

    }

    public String message(String systemMessage, String userMessage) {
        CompletionResponse response = gigachatClient.completions(CompletionRequest.builder()
                .model(ModelName.GIGA_CHAT_2)
                .functionCall("none")
                .messages(List.of(
                        ChatMessage.builder()
                                .content(systemMessage)
                                .role(ChatMessageRole.SYSTEM).build(),
                        ChatMessage.builder()
                                .content(userMessage)
                                .role(ChatMessageRole.USER)
                                .build()))
                .build());
        return response.choices().stream()
                .map(completionChoice -> completionChoice.message().content())
                .collect(Collectors.joining(". "));
    }


    public List<Embedding> embeddings(List<String> text) {
        EmbeddingResponse embeddingsQuery = gigachatClient.embeddings(EmbeddingRequest.builder()
                .model("EmbeddingsGigaR")
                .input(text)
                .build());
        return embeddingsQuery.data().stream()
                .map(e -> Embedding.from(e.embedding()))
                .toList();
    }

}
