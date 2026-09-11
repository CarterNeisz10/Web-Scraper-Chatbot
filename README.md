# Website Assistant

A Python-based website assistant that answers natural-language questions by semantically navigating websites, extracting relevant evidence, and generating grounded responses.

Rather than relying on predefined navigation paths or site-specific rules, the application uses sentence embeddings and cosine similarity to determine where to search. A bounded best-first search prioritizes relevant pages, while structured candidate detection and text-based evidence retrieval support both direct and generated answers.
 
## Demo

![Website Assistant Demo](assets/demo.gif)

Example queries:

```text
What is the price of the iPhone? https://www.apple.com
```

```text
What is the price of A Light in the Attic? https://books.toscrape.com
```

```text
What services does Prairie Sky View offer? https://www.prairieskyview.ca
```

---

## Overview

Website Assistant is a local desktop application designed to retrieve answers from unfamiliar websites using natural-language questions.

The system accepts a question and website URL, scrapes the website's visible content and links, and uses semantic similarity to determine which navigation paths are most relevant to the question. Relevant pages are explored using a priority queue until answer evidence is discovered or the configured search limit is reached.

The application supports multiple answer paths. Structured information such as prices can be extracted directly from webpage elements, multiple possible answers can trigger a clarification request, and general informational questions can be answered using text retrieved from a relevant destination page.

The goal of the project is to explore how semantic search, traditional web scraping, heuristic information extraction, and language models can be combined into a retrieval pipeline without requiring custom logic for each website.

## Features

- **Semantic link navigation** — Ranks webpage links against the user's question using sentence embeddings and cosine similarity.
- **Best-first website search** — Uses a priority queue to explore the most semantically relevant pages first.
- **Structured answer detection** — Identifies concrete values such as prices, percentages, and numerical values with units.
- **Candidate grouping** — Detects repeated webpage structures to identify groups of potential answers.
- **Clarification handling** — Requests additional information when multiple possible answers are discovered and semantically matches the user's clarification against those candidates.
- **Grounded text responses** — Generates answers using text retrieved from relevant webpages rather than relying on unrestricted model knowledge.
- **Generic website processing** — Uses semantic and structural heuristics rather than hardcoded company or product names.
- **Graphical and terminal interfaces** — Includes both a Tkinter desktop interface and a command-line interface for testing the retrieval pipeline.

## How It Works

The application processes a request through several stages:

1. **Parse the request**  
   The application separates the website URL from the user's natural-language question.

2. **Embed the question**  
   The question is converted into a semantic vector using `all-MiniLM-L6-v2`.

3. **Scrape the webpage**  
   The scraper retrieves visible page text, metadata, and navigation links using Requests and Beautiful Soup.

4. **Score navigation links**  
   Link text is embedded and compared with the question embedding using cosine similarity.

5. **Perform best-first navigation**  
   Relevant links are inserted into a priority queue. The crawler explores higher-scoring navigation paths before lower-scoring alternatives and stops after a bounded number of pages.

6. **Detect answer evidence**  
   The system searches for structured answer-bearing links and repeated candidate patterns. Price questions use stricter currency detection to avoid confusing unrelated numbers with prices.

7. **Resolve ambiguity**  
   If multiple candidates are found, the assistant asks the user to clarify. The clarification is embedded and compared only against the previously discovered candidate names.

8. **Use page-text evidence when appropriate**  
   For general informational questions, the system can identify when navigation has reached a relevant destination page and use its visible text as evidence.

9. **Generate the response**  
   Structured answers are returned directly, while text-based answers are generated with FLAN-T5 using only the retrieved website evidence.

## Architecture

The project separates scraping, retrieval, answer detection, and response generation into distinct components.

The processing pipeline is:

**User Question + URL → Request Processing → Semantic Link Ranking → Best-First Website Search → Evidence Detection → Answer or Clarification → Response**

The retrieval system intentionally separates **navigation relevance** from **answer detection**. Semantic similarity determines which pages should be searched, while structural and evidence-based heuristics determine whether the system has actually found an answer.

This prevents a semantically relevant navigation link from automatically being treated as the final answer.

### Search Strategy

The crawler uses semantic similarity as a navigation heuristic rather than treating it as proof that a link contains the answer.

Each available link is embedded with `all-MiniLM-L6-v2` and compared against the question embedding. Links above the configured relevance threshold are added to a priority queue. Because Python's `heapq` implements a min-heap, similarity scores are stored as negative values so that the highest-scoring links are explored first.

A visited-URL set prevents repeated page processing, while a maximum page count bounds the search.

### Candidate Detection

For structured questions, the application identifies links containing concrete answer-like values using regular expressions. Candidate links are converted into structural fingerprints by normalizing changing numerical values and comparing the remaining structure.

Jaccard similarity is used to detect repeated webpage patterns, helping distinguish collections of answer cards from isolated numerical or promotional links.

For price questions, candidate detection specifically requires currency values to reduce false positives from model numbers, years, and other unrelated numerical labels.

### Clarification

When multiple candidates are discovered, the assistant does not immediately select one.

Instead, it asks the user for clarification and semantically compares that response against the names of the existing candidates. A candidate is accepted only when the strongest cosine-similarity score reaches the configured clarification threshold.

If the clarification does not sufficiently match the existing candidates, the original question is refined and website search can continue.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Core application and search logic |
| **Sentence Transformers** | Semantic text embeddings |
| **all-MiniLM-L6-v2** | Question, link, and candidate embeddings |
| **FLAN-T5** | Grounded natural-language response generation |
| **Beautiful Soup** | HTML parsing and content extraction |
| **Requests** | HTTP webpage retrieval |
| **Tkinter** | Desktop graphical interface |
| **Regular Expressions** | Structured value and text-pattern detection |
| **heapq** | Priority queue for best-first website navigation |

## Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | Provides a terminal interface for testing the search and response pipeline. |
| `ui.py` | Implements the Tkinter chat interface and manages conversation state. |
| `brain.py` | Contains semantic navigation, best-first search, candidate detection, and clarification matching. |
| `scraper.py` | Retrieves webpages and extracts visible text, metadata, and links. |
| `response.py` | Processes requests and generates structured or evidence-grounded responses. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |

## Installation

### Prerequisites

- Python 3
- `pip`
- Internet access for retrieving webpages and downloading the required models on first use

### Clone the Repository

```bash
https://github.com/CarterNeisz10/Web-Scraper-Chatbot.git
cd Web-Scraper_Chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The required transformer models are downloaded automatically when the application is first run.

## Usage

### Graphical Interface

Run:

```bash
python ui.py
```

Enter a natural-language question followed by the website URL:

```text
What is the price of A Light in the Attic? https://books.toscrape.com
```

The application will search the website and display the resulting answer in the chat interface.

If multiple possible answers are detected, the assistant may request clarification before returning a final response.

### Terminal Interface

The retrieval pipeline can also be tested without the graphical interface:

```bash
python main.py
```

Then enter a question and URL when prompted.

## Example Behaviors

### Direct Structured Answer

```text
You:
What is the price of A Light in the Attic? https://books.toscrape.com

Assistant:
£51.77
```

The system semantically navigates from the bookstore homepage to the relevant book page and extracts the price.

### Clarification

```text
You:
What is the price of the iPhone? https://www.apple.com

Assistant:
Which iPhone option are you interested in?

You:
iPhone Air
```

When multiple relevant candidates are detected, the system asks the user to identify the intended option rather than arbitrarily selecting one.

### General Website Information

```text
You:
What services does Prairie Sky View offer? https://www.prairieskyview.ca
```

For non-price questions, the system can navigate to relevant website content and use the retrieved page text as grounded evidence for response generation.

## Design Decisions

### Semantic Navigation Instead of Site-Specific Rules

The crawler does not contain predefined navigation paths for individual websites. Navigation decisions are based on the semantic relationship between the user's question and the visible text of available links.

This allows the same retrieval pipeline to operate across websites with different structures without requiring custom selectors or product-specific navigation logic.

### Best-First Search

Relevant links are stored in a priority queue rather than followed strictly in page order. This allows the crawler to prioritize navigation paths that are semantically closer to the user's question while maintaining a fixed page-search limit.

### Evidence-Grounded Generation

The language model is used to transform retrieved website evidence into a concise answer for general informational questions. The generation prompt explicitly restricts the model to the supplied evidence rather than asking it to answer from unrestricted outside knowledge.

### Separate Navigation and Answer Detection

Semantic similarity determines **where the crawler should search**, while separate structural and evidence-based heuristics determine **whether an answer has been found**.

Keeping these responsibilities separate reduces the risk of treating a semantically relevant navigation link as the final answer.

## Limitations

Website Assistant is an experimental retrieval system and is not intended to successfully process every website.

Current limitations include:

- **JavaScript-rendered content** — The scraper uses standard HTTP requests and does not execute client-side JavaScript.
- **Anti-scraping protections** — Websites that block automated HTTP requests may not be accessible.
- **Complex website structures** — Sites with unusual navigation, minimal link text, or heavily dynamic interfaces may provide insufficient evidence for semantic navigation.
- **Heuristic structured extraction** — Candidate and value detection rely partly on structural and regular-expression heuristics and may not generalize to every webpage layout.
- **Limited crawl depth** — Website exploration is intentionally bounded to prevent uncontrolled crawling.
- **Local language model limitations** — Generated answers depend on both the quality of the retrieved evidence and the capabilities of the selected generation model.

These constraints are intentionally surfaced rather than hidden: the project focuses on demonstrating a generic semantic retrieval pipeline, not production-scale web indexing or browser automation.

## Future Improvements

Potential extensions include:

- Browser-based rendering for JavaScript-heavy websites
- More robust content extraction and page segmentation
- Improved semantic evidence retrieval within long webpages
- More sophisticated candidate and entity extraction
- Caching of previously processed pages and embeddings
- Expanded support for different structured answer types
- Improved confidence scoring and search termination
- Automated evaluation across a larger benchmark set
