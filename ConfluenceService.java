package org.example.agent;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.example.model.ConfluenceRequest;
import org.example.model.ConfluenceResponse;
import org.jsoup.nodes.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class ConfluenceService {

    private static final Logger log = LoggerFactory.getLogger(ConfluenceService.class);
    private static final String promptPattern = "Используйте только следующую информацию: %s и не используй свои знания не изменяй названия, " +
            "отделяй слова пробелами"; 
    
    private final EmbeddingService embeddingService;
    private final GigaChatClientService gigaChatClientService;
    private final UrlDocumentService urlDocumentService;

    public ConfluenceService(EmbeddingService embeddingService,
                             GigaChatClientService gigaChatClientService,
                             UrlDocumentService urlDocumentService) {
        this.embeddingService = embeddingService;
        this.gigaChatClientService = gigaChatClientService;
        this.urlDocumentService = urlDocumentService;
    }

    public ConfluenceResponse getConfluenceResponse(ConfluenceRequest request) {
        String url = request.parentUrl();
        Integer depth = request.depth();
        Set<String> links = urlDocumentService.findLinks(url, depth);
        StringBuilder sbAllDescription = new StringBuilder();
        Set<String> processedLinks = new HashSet<>(); // Processed tasks
        links.forEach(link -> {
            Document doc = urlDocumentService.getUrlDocument(link);
            if (doc == null) {
                return;
            }
            String description = urlDocumentService.getDescription(link, doc);
            if (description != null) {
                sbAllDescription.append(description).append(".\n");
                processedLinks.add(link);
                log.info(link);
            }
        });

        String query = request.query();
        try (InputStream allDescriptionStream = new ByteArrayInputStream(sbAllDescription.toString().getBytes())) {
            List<TextSegment> textSegments = embeddingService.documentSplitt(allDescriptionStream);
            EmbeddingStore<TextSegment> textSegmentEmbeddingStore = embeddingService.embeddingToStore(textSegments);
            Embedding embedQuery = embeddingService.embed(query);
            List<EmbeddingMatch<TextSegment>> embeddingMatches = embeddingService.queryEmbedding(embedQuery, textSegmentEmbeddingStore);
            List<TextSegment> answerOverlap = embeddingService.queryEmbeddingOverlap(textSegments, embeddingMatches);
            //
            String matchText = answerOverlap.stream()
                    .map(TextSegment::text)
                    .collect(Collectors.joining(" "));
            log.info("Match text= {}", matchText);
            String prompt = String.format(promptPattern, matchText);
            String responseMessage = gigaChatClientService.message(prompt, query +"?");
            log.info("message:{}", responseMessage);
            return new ConfluenceResponse(responseMessage, processedLinks.stream().toList());
        } catch (Exception e) {
            log.error("TextDocumentParser error", e);
            return new ConfluenceResponse(e.getMessage(), Collections.emptyList());
        }
    }

}
