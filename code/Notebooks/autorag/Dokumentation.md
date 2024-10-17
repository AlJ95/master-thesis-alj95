# AutoRAG 

Ziel ist es herauszufinden, wie nutzerfreundlich die Frameworks sind, welche Funktionalitäten und Eigenschaften ich übernehmen sollte und herauszufinden, ob diese auch so variable einsetzbar sind, wie mein Vorhaben.

Ich teste dies an dem SE4AI Datensatz Mietgesetze und teste, ob ich damit ein RAG mit fortgeschrittener IRA Methode umsetzen kann.

## Fazit

Die Github Repo ist an sich gut dokumentiert, aber die Datenvorgaben waren nicht eindeutig
-> Alle UI Elemente müssen eindeutig dokumentiert sein.

Mit diesem Framework lassen sich, laut Discord, derzeit keine IRA Methoden umsetzen.

Das Framework bietet an sich eine sehr breite Auswahl von RAG Komponenten

Es ist eher eine generalisierte Evaluation als eine Task-Spezifische. Ich habe keine Möglichkeiten gesehen, eigene Metriken zu entwickeln -> Dafür müsste es geforked werden.

## LOGGING

> (.venv-autorag) janalbrecht@MacBook-Air-von-Admin master-thesis-alj95 % python code/Notebooks/autorag/autorag-test.py
> None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
> [10/17/24 17:27:27] INFO     [__init__.py:99] >> You are using API version of AutoRAG.To use local version, run pip install 'AutoRAG[gpu]'  __init__.py:99
> INFO     [__init__.py:117] >> You are using API version of AutoRAG.To use local version, run pip install               __init__.py:117
>          'AutoRAG[gpu]'                                                                                                               
> [10/17/24 17:27:31] INFO     [evaluator.py:165] >> Embedding BM25 corpus...                                                               evaluator.py:165
> INFO     [evaluator.py:185] >> BM25 corpus embedding complete.                                                        evaluator.py:185
> INFO     [posthog.py:20] >> Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry   posthog.py:20
>          for more information.                                                                                                        
> [10/17/24 17:27:32] INFO     [evaluator.py:229] >> Embedding VectorDB corpus with openai_embed_3_small...                                 evaluator.py:229
> INFO     [evaluator.py:251] >> VectorDB corpus embedding complete with openai_embed_3_small.                          evaluator.py:251
> INFO     [evaluator.py:143] >> Running node line retrieve_node_line...                                                evaluator.py:143
> INFO     [node.py:55] >> Running node retrieval...                                                                          node.py:55
> INFO     [run.py:166] >> Running retrieval node - semantic retrieval module...                                              run.py:166
> INFO     [base.py:18] >> Initialize retrieval node - VectorDB                                                               base.py:18
> INFO     [base.py:31] >> Running retrieval node - VectorDB module...                                                        base.py:31
> [10/17/24 17:27:33] INFO     [_client.py:1038] >> HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"                _client.py:1038
> [10/17/24 17:27:36] INFO     [base.py:28] >> Deleting retrieval node - VectorDB module...                                                       base.py:28
> INFO     [run.py:197] >> Running retrieval node - lexical retrieval module...                                               run.py:197
> INFO     [base.py:18] >> Initialize retrieval node - BM25                                                                   base.py:18
> [10/17/24 17:27:37] INFO     [base.py:31] >> Running retrieval node - BM25 module...                                                            base.py:31
> INFO     [base.py:28] >> Deleting retrieval node - BM25 module...                                                           base.py:28
> INFO     [run.py:228] >> Running retrieval node - hybrid retrieval module...                                                run.py:228
> INFO     [base.py:18] >> Initialize retrieval node - VectorDB                                                               base.py:18
> INFO     [base.py:31] >> Running retrieval node - VectorDB module...                                                        base.py:31
> [10/17/24 17:27:38] INFO     [_client.py:1038] >> HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"                _client.py:1038
> INFO     [posthog.py:20] >> Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry   posthog.py:20
>          for more information.                                                                                                        
> [10/17/24 17:27:39] INFO     [base.py:28] >> Deleting retrieval node - VectorDB module...                                                       base.py:28
> INFO     [base.py:18] >> Initialize retrieval node - BM25                                                                   base.py:18
> INFO     [base.py:31] >> Running retrieval node - BM25 module...                                                            base.py:31
> INFO     [base.py:28] >> Deleting retrieval node - BM25 module...                                                           base.py:28
> 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77/77 [00:15<00:00,  4.88it/s]
> [10/17/24 17:27:55] INFO     [evaluator.py:143] >> Running node line post_retrieve_node_line...                                           evaluator.py:143
> INFO     [node.py:55] >> Running node prompt_maker...                                                                       node.py:55
> INFO     [base.py:15] >> Initialize prompt maker node - Fstring module...                                                   base.py:15
> INFO     [base.py:23] >> Running prompt maker node - Fstring module...                                                      base.py:23
> INFO     [base.py:20] >> Prompt maker node - Fstring module is deleted.                                                     base.py:20
> [10/17/24 17:27:56] INFO     [node.py:55] >> Running node generator...                                                                          node.py:55
> INFO     [base.py:19] >> Initialize generator node - OpenAILLM                                                              base.py:19
> [10/17/24 17:27:57] INFO     [base.py:26] >> Running generator node - OpenAILLM module...                                                       base.py:26
> [10/17/24 17:28:00] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:01] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:02] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:03] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:04] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:06] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:11] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> [10/17/24 17:28:13] INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [_client.py:1786] >> HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"          _client.py:1786
> INFO     [base.py:23] >> Deleting generator module - OpenAILLM                                                              base.py:23
> [nltk_data] Downloading package punkt_tab to
> [nltk_data]     /Users/janalbrecht/nltk_data...
> [nltk_data]   Package punkt_tab is already up-to-date!
> [nltk_data] Downloading package wordnet to
> [nltk_data]     /Users/janalbrecht/nltk_data...
> [nltk_data] Downloading package punkt_tab to
> [nltk_data]     /Users/janalbrecht/nltk_data...
> [nltk_data]   Package punkt_tab is already up-to-date!
> [nltk_data] Downloading package omw-1.4 to
> [nltk_data]     /Users/janalbrecht/nltk_data...
> Generating embeddings:   0%|                                                                                                       | 0/21 [00:00<?, ?it/s][10/17/24 17:29:19] INFO     [_client.py:1038] >> HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"                _client.py:1038
> Generating embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:01<00:00, 11.19it/s]
> Generating embeddings:   0%|                                                                                                       | 0/21 [00:00<?, ?it/s][10/17/24 17:29:20] INFO     [_client.py:1038] >> HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"                _client.py:1038
> Generating embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:01<00:00, 17.95it/s]
> [10/17/24 17:29:21] INFO     [evaluator.py:154] >> Evaluation complete.   