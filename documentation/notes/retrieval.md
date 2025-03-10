## Zusammenfassung: Haystack Retrieval Evaluierung, LLM-as-a-Judge und Preference Leakage

**Haystack und Retrieval Evaluierung mit LLM-as-a-Judge:**

Haystack bietet auch Integrationen mit externen Frameworks zur Evaluierung von RAG Pipelines, wie z.B. DeepEval, RAGAS und UpTrain. ([Haystack Dokumentation zu Evaluatoren](https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines))

Haystack integriert sich mit externen Frameworks, die speziell für die Evaluierung von RAG-Systemen entwickelt wurden.  Zu diesen gehören:

*   **RAGAS:** RAGAS (Retrieval-Augmented Generation Assessment) ist ein Framework zur Evaluierung der Qualität von RAG-Pipelines. Es bietet Metriken wie Kontextrelevanz, Antworttreue und Antwortrelevanz, um verschiedene Aspekte der RAG-Pipeline-Leistung zu bewerten.  Haystack bietet Integrationen mit RAGAS, um dessen Metriken in Evaluationsworkflows zu nutzen. ([Haystack Dokumentation zu Evaluatoren](https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines) im File `documentation/notes/retrieval.md`)
*   **DeepEval:** DeepEval ist ein weiteres Framework für die Evaluierung von LLM-basierten Systemen, einschließlich RAG-Pipelines.  Es bietet eine Reihe von Metriken und Evaluatoren, die auf verschiedenen Aspekten der Textqualität und Relevanz basieren. Haystack integriert sich mit DeepEval, um dessen Evaluationsfähigkeiten in Haystack Pipelines zu integrieren. ([Haystack Dokumentation zu Evaluatoren](https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines) im File `documentation/notes/retrieval.md`)
*   **UpTrain:** UpTrain ist eine Plattform für das Monitoring und die Evaluation von KI-Systemen, einschließlich RAG-Pipelines.  Es bietet Tools zur Verfolgung der Pipeline-Leistung im Laufe der Zeit und zur Identifizierung von Bereichen, die verbessert werden müssen. Haystack bietet Integrationen mit UpTrain, um dessen Monitoring- und Evaluationsfunktionen zu nutzen. ([Haystack Dokumentation zu Evaluatoren](https://haystack.deepset.ai/tutorials/35_evaluating_rag_pipelines) im File `documentation/notes/retrieval.md`)

**Prometheus 2 und LLM-as-a-Judge:**

Prometheus (genauer Prometheus v2) wird im Kontext von LLM-basierter Evaluation als "LLM-as-a-Judge" Modell erwähnt.  Diese Modelle, wie Prometheus v2, werden trainiert, um die Qualität von LLM-generierten Texten zu bewerten, ähnlich wie ein menschlicher Gutachter.  Prometheus v2 wurde mit Feedback-Daten von GPT-4 trainiert und dient als "fine-grained Evaluator".  Solche LLM-as-a-Judge Modelle können in Evaluationspipelines eingesetzt werden, um automatisiert die Qualität von RAG-Antworten zu beurteilen.  Haystack ermöglicht die Integration solcher Modelle in Evaluationsworkflows, obwohl die explizite Integration von Prometheus v2 in den Web-Suchergebnissen oder der beigefügten Datei nicht detailliert beschrieben ist.  Es ist jedoch wahrscheinlich, dass Haystack die Nutzung von Hugging Face Modellen wie Prometheus v2 über seine generischen Integrationsmöglichkeiten unterstützt. ([arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) im File `documentation/notes/retrieval.md`)


**LLM-as-a-Judge Modelle und Teacher-Modelle:**

In der Forschung werden verschiedene LLMs als Judges untersucht. Man unterscheidet zwischen Closed-Source (proprietär) und Open-Source (fine-tuned) Modellen:

**Closed-Source Modelle:**

*   **GPT-4, ChatGPT, Claude:**  Diese Modelle von OpenAI und Anthropic gelten als sehr leistungsfähig und werden oft als Goldstandard für LLM-as-a-Judge Evaluierungen verwendet. Sie zeigen hohe Übereinstimmung mit menschlichen Bewertungen. ([Zheng et al., 2023c in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1), [Zhou et al., 2023 in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))

**Open-Source Modelle (Fine-tuned):**

Diese Modelle werden durch Fine-Tuning von Open-Source LLMs mit dem Ziel trainiert, die Bewertungsfähigkeiten von starken Closed-Source Modellen zu imitieren (Knowledge Distillation).  Oftmals ist **GPT-4** das Teacher-Modell, von dem diese Modelle lernen. Beispiele sind:

*   **JudgeLM:**  Ein fine-tuned Vicuna Modell, trainiert mit GPT-4 generierten Bewertungen. Teacher-Modell: **GPT-4**. ([JudgeLM GitHub](https://github.com/baaivision/JudgeLM), [Zhu et al., 2023a in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))
*   **Themis:**  Ein fine-tuned LLM Judge, ebenfalls trainiert, um die Fähigkeiten von **GPT-4** zu destillieren. Teacher-Modell: **GPT-4**. ([Hu et al., 2025 in arXiv:2502.02988](https://arxiv.org/html/2502.02988))
*   **Prometheus:**  Ein fine-grained Evaluator, trainiert mit Feedback-Daten von **GPT-4**. Teacher-Modell: **GPT-4**. ([Kim et al., 2023 in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))
*   **PandaLM:**  Trainiert mit Annotationen von **GPT-3.5**. Teacher-Modell: **GPT-3.5**. ([Wang et al., 2023d in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))
*   **SelFee:**  Basiert auf Feedback von **ChatGPT**. Teacher-Modell: **ChatGPT**. ([Ye et al., 2023 in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))
*   **Fine-tuned Vicuna:**  Feinabstimmung von Vicuna, oft unter Nutzung von Datensätzen, die mit starken Modellen wie **GPT-4 oder ChatGPT** erstellt wurden. Indirekte Teacher-Modelle: **GPT-4 oder ChatGPT**. ([Zheng et al., 2023d in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))
*   **Shepherd:**  Basiert primär auf Community- und Human-Feedback, kein explizites LLM Teacher-Modell, aber indirekt könnten starke LLMs bei der Datenerstellung geholfen haben. ([Wang et al., 2023c in arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1))

**Preference Leakage:**

"Preference Leakage" ist ein Kontaminationsproblem, das auftreten kann, wenn verwandte LLMs sowohl für die Generierung synthetischer Daten als auch für die Evaluierung (als Judge) verwendet werden.  Es beschreibt die Tendenz von Judge-LLMs, "Studentenmodelle" (Modelle, die mit den synthetischen Daten des Generator-LLMs trainiert wurden) **ungerechtfertigt positiv zu bewerten**, wenn eine Verwandtschaft zwischen Generator und Judge besteht.  Verwandtschaft kann bedeuten:

1.  **Gleiches Modell**
2.  **Vererbungsbeziehung**
3.  **Gleiche Modellfamilie**

"Preference Leakage" ist ein schwer zu entdeckendes Problem und kann die Validität von LLM-Evaluierungen beeinträchtigen. ([Li et al., 2025 in arXiv:2502.01534](https://arxiv.org/abs/2502.01534))


## Quellen

Okay, hier sind die direkten Zitate aus den Quellen, die die Teacher-Modelle für die Open-Source LLM-as-a-Judge Modelle belegen. Ich habe versucht, so präzise wie möglich zu sein in Bezug auf die Quellenangabe:

**JudgeLM:**

*   "JudgeLM dataset contains 100K judge samples for training and 5K judge samples for validation. All the judge samples have the **GPT-4-generated high-quality judgements**." -> [JudgeLM GitHub](https://github.com/baaivision/JudgeLM) -  Absatz "Overview", unter "JudgeLM dataset contains..." ✅ 

**Themis:**

*   "These designs enable Themis to effectively distill evaluative skills from **teacher models**, while retaining flexibility for continuous development. ... The results reveal that Themis achieves comparable and slightly worse performance on the in- and out-of-distribution benchmarks, respectively, using less than 1% of parameters compared with its **teacher GPT-4**, and outperforms all other tested (judge) LLMs." -> [arXiv:2502.02988](https://arxiv.org/html/2502.02988) -  Absatz 2 des Abstracts ✅ 


**Prometheus:**

*   "Prometheus (Kim et al., 2023) defines thousands of evaluation criteria and construct a feedback dataset based on **GPT-4**, and fine-tunes a fine-grained evaluator model." -> [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) - Abschnitt 2.2.2 "Fine-tuned LLM", Absatz 1, Satz 3 ✅ 

**PandaLM:**

*   "PandaLM (Wang et al., 2023d) constructs data based on Alpaca instructions and **GPT-3.5 annotation**, and then fine-tunes LLaMA-7B (Touvron et al., 2023a) as an evaluator model." -> [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) - Abschnitt 2.2.2 "Fine-tuned LLM", Absatz 1, Satz 2 ✅ 

**SelFee:**

*   "SelFee (Ye et al., 2023) collects generations, feedback, and revised generations from **ChatGPT** and fine-tunes LLaMA models to build a critique model." -> [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) - Abschnitt 2.4.1 "LLM-as-a-Judge for model", Absatz 2, Satz 2

**Fine-tuned Vicuna:**

*   "Zheng et al. (2023d) also fine-tune Vicuna (Touvron et al., 2023b) on a 20K pairwise comparison dataset to explore the potential of open-source models as a more cost-friendly proxy." -> [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) - Abschnitt 2.4.1 "LLM-as-a-Judge for model", Absatz 2, Satz 5
    *   *Anmerkung:* Das Paper [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) erwähnt nicht explizit das Teacher-Modell für die Erstellung des 20K Pairwise Comparison Datensatzes für Vicuna. Es ist jedoch gängige Praxis, für solche Datensätze starke Modelle wie **GPT-4 oder ChatGPT** zu verwenden.  Das Paper [arXiv:2502.02988](https://arxiv.org/html/2502.02988) impliziert, dass Vicuna zu den "established judges" gehört, die "general-purpose LLMs (Lin et al., 2024a; Zheng et al., 2023) or collect real user instructions to construct their fine-tuning datasets (Li et al., 2023; Ke et al., 2024)" nutzen, was auf eine Nutzung von starken LLMs bei der Datenerstellung hindeutet.*

**Shepherd:**

*   *Für Shepherd konnte in den bereitgestellten Dokumenten kein direktes Zitat gefunden werden, das ein explizites LLM als Teacher-Modell nennt.*  Das Paper [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) beschreibt Shepherd wie folgt: "Shepherd (Wang et al., 2023c) trains a model that can output critiques for single-response with the data of feedback from online communities and human annotation."  Dies deutet darauf hin, dass **menschliches Feedback und Daten aus Online-Communities** die primären Trainingsdatenquellen sind, und kein LLM explizit als Teacher-Modell fungiert.  Indirekt könnten LLMs jedoch bei der Sammlung oder Aufbereitung dieser Feedback-Daten eine Rolle gespielt haben.

Ich hoffe, diese detaillierte Liste mit Zitaten ist hilfreich für dich! Lass mich wissen, wenn du noch etwas benötigst.


 Haystack integriert externe Frameworks zur RAG-Evaluierung wie DeepEval, RAGAS und UpTrain.
    ```
    "Haystack bietet auch Integrationen mit externen Frameworks zur Evaluierung von RAG Pipelines, wie z.B. DeepEval, RAGAS und UpTrain." -> aus `documentation/notes/retrieval.md` unter Kapitel "Haystack und Retrieval Evaluierung mit LLM-as-a-Judge" Absatz 1
    ```
*   RAGAS ist ein Framework zur Bewertung von RAG-Pipelines mit Metriken wie Kontextrelevanz und Antworttreue.
    ```
    "RAGAS (Retrieval-Augmented Generation Assessment) ist ein Framework zur Evaluierung der Qualität von RAG-Pipelines." ->  basierend auf allgemeinem Wissen über RAGAS und dessen Zweck (nicht direkt zitiert in den Quellen, aber implizit durch die Erwähnung in `documentation/notes/retrieval.md`)
    ```
*   Prometheus ist ein "fine-grained Evaluator", der mit GPT-4 Feedback-Daten trainiert wurde.
    ```
    "Prometheus (Kim et al., 2023) defines thousands of evaluation criteria and construct a feedback dataset based on GPT-4, and fine-tunes a fine-grained evaluator model." -> aus [arXiv:2411.15594v1](https://arxiv.org/html/2411.15594v1) im File `documentation/notes/retrieval.md` unter Kapitel "Quellen" -> "Prometheus"
    ```
*   Haystack bietet `DoclingConverter` für verschiedene Dokumenttypen und `ExportType` Optionen für RAG.
    ```
    "The presented `DoclingConverter` component enables you to: ... use various document types in your LLM applications with ease and speed, and leverage Docling's rich format for advanced, document-native grounding." -> aus [https://ds4sd.github.io/docling/examples/rag_haystack/](https://ds4sd.github.io/docling/examples/rag_haystack/) unter Kapitel "Overview" Absatz 2
    ```
*   `DoclingConverter` unterstützt `ExportType.MARKDOWN` und `ExportType.DOC_CHUNKS`.
    ```
    "`DoclingConverter` supports two different export modes: * `ExportType.MARKDOWN`: if you want to capture each input document as a separate Haystack document, or * `ExportType.DOC_CHUNKS` (default): if you want to have each input document chunked and to then capture each individual chunk as a separate Haystack document downstream." -> aus [https://ds4sd.github.io/docling/examples/rag_haystack/](https://ds4sd.github.io/docling/examples/rag_haystack/) unter Kapitel "Overview" Absatz 3



## RAGAS


### The Problem with RAGAS Context Precision

RAGAS provides a useful metric called `LLMContextPrecisionWithoutReference` for evaluating retrieval quality. However, there's a fundamental issue with how this metric works: **it evaluates contexts based on their relevance to the generated response rather than the original query**.

Looking at the RAGAS documentation:

```python
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."], 
)

await context_precision.single_turn_ascore(sample)
```

Notice how the RAGAS metric compares each retrieved context against the `response`. This approach has a significant flaw:

1. It measures **context utilization** (how well the generator used the contexts) rather than **context relevance** (how well the retriever found relevant contexts)
2. A perfectly irrelevant context that happens to match the response will score high
3. A highly relevant context that wasn't used in the response will score low

### Our Solution: Query-Based Context Precision

We've implemented a custom `LLMContextPrecisionMetric` that properly evaluates retrieval by judging context relevance against the original query and expected answer, not the generated response.

Here's how our implementation works:

1. For each query and its retrieved contexts, we ask an LLM to judge if each context is relevant to answering the query
2. We calculate precision as (number of relevant contexts) / (total number of contexts)
3. This gives us a true measure of retrieval quality independent of how the generator used the contexts

This approach aligns with the fundamental purpose of a retriever: finding contexts that contain information relevant to answering the user's query.

### Benefits of Our Approach

1. **True retrieval evaluation**: Measures how well the retriever finds relevant contexts for the query
2. **Independence from generator**: Evaluation isn't affected by how the generator uses the contexts

By focusing on the relevance of contexts to the query rather than to the response, we get a more accurate picture of retriever performance, which better guides retrieval optimization efforts.
