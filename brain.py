import heapq

from sentence_transformers import SentenceTransformer, util
from scraper import scrape_website


# --------------------------------------------------
# Model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

CONTENT_THRESHOLD = 0.55
LINK_THRESHOLD = 0.40

# How far ahead the best candidate must be before
# we consider it clearly better than alternatives.
ANSWER_MARGIN = 0.08

MAX_PAGES = 10

# Number of evidence candidates remembered.
MAX_CANDIDATES = 10


# --------------------------------------------------
# Text processing
# --------------------------------------------------

def split_text(text, chunk_size=100):
    """
    Splits page text into chunks used as potential
    answer evidence.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Evidence candidates
# --------------------------------------------------

def find_content_candidates(
    question_embedding,
    page_text,
    page_url
):
    """
    Scores every text chunk on the page and returns
    chunks that are sufficiently related to the
    user's question.
    """
    chunks = split_text(page_text)

    if not chunks:
        return []

    chunk_embeddings = embedding_model.encode(
        chunks
    )

    scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    candidates = []

    for index, chunk in enumerate(chunks):

        score = scores[index].item()

        if score >= CONTENT_THRESHOLD:

            candidates.append({
                "text": chunk,
                "source_url": page_url,
                "similarity": score
            })

    candidates.sort(
        key=lambda candidate:
        candidate["similarity"],
        reverse=True
    )

    return candidates


def add_candidates(
    all_candidates,
    new_candidates
):
    """
    Adds newly discovered evidence while avoiding
    duplicate text.
    """
    existing_text = {
        candidate["text"]
        for candidate in all_candidates
    }

    for candidate in new_candidates:

        if candidate["text"] not in existing_text:

            all_candidates.append(
                candidate
            )

            existing_text.add(
                candidate["text"]
            )

    all_candidates.sort(
        key=lambda candidate:
        candidate["similarity"],
        reverse=True
    )

    return all_candidates[
        :MAX_CANDIDATES
    ]


# --------------------------------------------------
# Link scoring
# --------------------------------------------------

def find_relevant_links(
    question_embedding,
    links,
    visited_urls
):
    """
    Scores links against the user's question.
    """
    usable_links = []

    seen_urls = set()

    for link in links:

        text = link["text"].strip()
        url = link["url"]

        if not text:
            continue

        if url in visited_urls:
            continue

        # Avoid putting the same URL into this
        # result several times.
        if url in seen_urls:
            continue

        seen_urls.add(url)

        usable_links.append(link)

    if not usable_links:
        return []

    link_texts = [
        link["text"]
        for link in usable_links
    ]

    link_embeddings = embedding_model.encode(
        link_texts
    )

    scores = util.cos_sim(
        question_embedding,
        link_embeddings
    )[0]

    relevant_links = []

    for index, link in enumerate(
        usable_links
    ):

        score = scores[index].item()

        if score >= LINK_THRESHOLD:

            relevant_links.append({
                "text": link["text"],
                "url": link["url"],
                "similarity": score
            })

    relevant_links.sort(
        key=lambda link:
        link["similarity"],
        reverse=True
    )

    return relevant_links


# --------------------------------------------------
# Candidate decision
# --------------------------------------------------

def evaluate_candidates(candidates):
    """
    Examines the evidence gathered during the search.

    Returns:

        found
        needs_clarification
        not_found
    """

    if not candidates:

        return {
            "status": "not_found"
        }

    best = candidates[0]

    # Only one viable answer candidate exists.
    if len(candidates) == 1:

        return {
            "status": "found",
            "evidence": best["text"],
            "source_url":
                best["source_url"],
            "similarity":
                best["similarity"]
        }

    second_best = candidates[1]

    margin = (
        best["similarity"]
        - second_best["similarity"]
    )

    # One candidate clearly dominates.
    if margin >= ANSWER_MARGIN:

        return {
            "status": "found",
            "evidence": best["text"],
            "source_url":
                best["source_url"],
            "similarity":
                best["similarity"]
        }

    # Several candidates are similarly plausible.
    return {
        "status":
            "needs_clarification",

        "reason":
            "multiple relevant answers were found",

        "options":
            candidates[:5]
    }


# --------------------------------------------------
# Main search
# --------------------------------------------------

def search_website(
    starting_url,
    question_embedding
):
    """
    Best-First Search.

    Rather than accepting the first relevant chunk,
    evidence is collected while the website is
    searched.

    Final possibilities:

        found
        needs_clarification
        not_found
    """

    visited_urls = set()

    queued_urls = {
        starting_url
    }

    pages_to_visit = []

    all_candidates = []

    heapq.heappush(
        pages_to_visit,
        (-1.0, starting_url)
    )

    pages_visited = 0

    # --------------------------------
    # Search
    # --------------------------------

    while (
        pages_to_visit
        and pages_visited < MAX_PAGES
    ):

        negative_score, current_url = (
            heapq.heappop(
                pages_to_visit
            )
        )

        queued_urls.discard(
            current_url
        )

        if current_url in visited_urls:
            continue

        visited_urls.add(
            current_url
        )

        pages_visited += 1

        print(
            f"\nVisiting: {current_url}"
        )

        # --------------------------------
        # Scrape
        # --------------------------------

        page = scrape_website(
            current_url
        )

        if page is None:

            print(
                "Scraper could not retrieve page."
            )

            continue

        # --------------------------------
        # Gather evidence
        # --------------------------------

        page_candidates = (
            find_content_candidates(
                question_embedding,
                page["text"],
                page["url"]
            )
        )

        if page_candidates:

            print(
                "Best content similarity: "
                f"{page_candidates[0]['similarity']:.3f}"
            )

            all_candidates = add_candidates(
                all_candidates,
                page_candidates
            )

        else:

            print(
                "No strong content candidate "
                "on this page."
            )

        # --------------------------------
        # Find navigation options
        # --------------------------------

        relevant_links = (
            find_relevant_links(
                question_embedding,
                page["links"],
                visited_urls
            )
        )

        if relevant_links:

            print("Promising links:")

            for link in relevant_links[:5]:

                print(
                    f"  {link['text']} "
                    f"({link['similarity']:.3f})"
                )

        else:

            print(
                "No sufficiently relevant "
                "links on this page."
            )

        # --------------------------------
        # Queue links
        # --------------------------------

        for link in relevant_links:

            url = link["url"]

            if url in visited_urls:
                continue

            if url in queued_urls:
                continue

            heapq.heappush(
                pages_to_visit,
                (
                    -link["similarity"],
                    url
                )
            )

            queued_urls.add(url)

    # --------------------------------
    # Search finished
    # --------------------------------

    print(
        "\nEvaluating gathered evidence..."
    )

    if all_candidates:

        print("\nTop candidates:")

        for candidate in all_candidates[:5]:

            preview = (
                candidate["text"][:80]
                .replace("\n", " ")
            )

            print(
                f"  {candidate['similarity']:.3f} "
                f"- {preview}..."
            )

    result = evaluate_candidates(
        all_candidates
    )

    print(
        f"\nBrain decision: "
        f"{result['status']}"
    )

    return result
