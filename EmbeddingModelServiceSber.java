package org.example.agent;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import jakarta.annotation.PostConstruct;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@Profile("sber")
public class EmbeddingModelServiceSber implements EmbeddingModelService {

    private final GigaChatClientService gigachatClient;

    public EmbeddingModelServiceSber(GigaChatClientService gigachatClient) {
        this.gigachatClient = gigachatClient;
    }

    @Override
    public List<Embedding> embedAll(List<TextSegment> segments) {
        return gigachatClient.embeddings(segments.stream()
                .map(TextSegment::text)
                .collect(Collectors.toList()));
    }

    @Override
    public Embedding embed(String text) {
        return gigachatClient.embeddings(List.of(text)).get(0);
    }

    @Override
    public int dimension() {
        return this.embed("test").dimension();
    }

}
