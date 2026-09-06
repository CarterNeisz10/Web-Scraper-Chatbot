import heapq
import re

from sentence_transformers import SentenceTransformer, util
from scraper import scrape_website


# --------------------------------------------------
# Model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Search settings
# --------------------------------------------------

LINK_THRESHOLD = 0.40
MAX_PAGES = 10
CLARIFICATION_THRESHOLD = 0.90

# --------------------------------------------------
# Link scoring
# --------------------------------------------------

def score_links(
    question_embedding,
    links,
    visited_urls
):
    """
    Scores links for navigation.

    Similarity is used to decide WHERE to search,
    not which answer candidate the user means.
    """

    usable_links = [
        link
        for link in links
        if link["text"].strip()
        and link["url"] not in visited_urls
    ]

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

    scored_links = []

    for index, link in enumerate(
        usable_links
    ):

        scored_links.append({
            "text":
                link["text"],

            "url":
                link["url"],

            "similarity":
                scores[index].item()
        })

    scored_links.sort(
        key=lambda link:
        link["similarity"],
        reverse=True
    )

    return scored_links


# --------------------------------------------------
# Remove duplicate links
# --------------------------------------------------

def remove_duplicate_links(links):

    unique_links = []
    seen = set()

    for link in links:

        key = (
            link["text"]
            .strip()
            .lower(),

            link["url"]
        )

        if key in seen:
            continue

        seen.add(key)

        unique_links.append(
            link
        )

    return unique_links


# --------------------------------------------------
# Detect answer-bearing links
# --------------------------------------------------

def contains_answer_value(text):
    """
    Detects links containing concrete values.

    Examples:

        $699
        15%
        24 hours
        256 GB
        3 years

    No company/product names are hardcoded.
    """

    patterns = [

        # Currency
        r"[$€£¥]\s*[\d,.]+",

        # Percentage
        r"\b\d+(?:\.\d+)?\s*%",

        # Number + unit/word
        r"\b\d+(?:\.\d+)?\s+"
        r"[A-Za-z]+"
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            return True

    return False


# --------------------------------------------------
# Detect standalone answer values
# --------------------------------------------------

def is_standalone_answer_value(text):
    """
    Detects when a link is essentially just one
    concrete value rather than promotional text.

    This lets a strongly relevant page return a
    single direct answer without requiring a
    repeated candidate-card pattern.
    """

    clean_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    patterns = [
        r"[$€£¥]\s*[\d,.]+",
        r"\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s+[A-Za-z]+"
    ]

    return any(
        re.fullmatch(
            pattern,
            clean_text,
            re.IGNORECASE
        )
        for pattern in patterns
    )


# --------------------------------------------------
# Structural fingerprint
# --------------------------------------------------

def get_link_structure(text):
    """
    Creates a rough structural fingerprint.

    The actual words and numbers are removed so
    repeated answer-card patterns can be detected.

    Example:

        Product A ... Buy from $699 Buy
        Product B ... Buy from $799 Buy

    should produce similar fingerprints.
    """

    structure = text.lower()

    # Replace currency values
    structure = re.sub(
        r"[$€£¥]\s*[\d,.]+(?:\s*[–-]\s*[$€£¥]?\s*[\d,.]+)?",
        " VALUE ",
        structure
    )

    # Replace other numbers
    structure = re.sub(
        r"\d+(?:\.\d+)?",
        " NUMBER ",
        structure
    )

    # Remove words that are likely unique names.
    # We preserve the later repeated action/value
    # structure by looking at the end of the text.
    words = structure.split()

    if len(words) > 12:
        words = words[-12:]

    structure = " ".join(
        words
    )

    # Normalize whitespace
    structure = re.sub(
        r"\s+",
        " ",
        structure
    ).strip()

    return structure


# --------------------------------------------------
# Structure similarity
# --------------------------------------------------

def structure_similarity(
    structure_a,
    structure_b
):
    """
    Simple word-overlap comparison between two
    structural fingerprints.
    """

    words_a = set(
        structure_a.split()
    )

    words_b = set(
        structure_b.split()
    )

    if not words_a or not words_b:
        return 0.0

    intersection = (
        words_a.intersection(
            words_b
        )
    )

    union = (
        words_a.union(
            words_b
        )
    )

    return (
        len(intersection)
        /
        len(union)
    )


# --------------------------------------------------
# Find answer-bearing links
# --------------------------------------------------

def get_answer_links(scored_links):
    """
    Gets every link containing concrete answer-like
    information.

    IMPORTANT:

    There is NO semantic candidate threshold here.
    """

    answer_links = []

    for link in scored_links:

        if contains_answer_value(
            link["text"]
        ):

            candidate = dict(link)

            candidate["structure"] = (
                get_link_structure(
                    link["text"]
                )
            )

            answer_links.append(
                candidate
            )

    return answer_links


# --------------------------------------------------
# Find repeated candidate group
# --------------------------------------------------

def find_candidate_group(
    scored_links
):
    """
    Finds the largest group of answer-bearing links
    sharing a repeated page structure.

    This helps separate a repeated set of actual
    answer cards from one-off promotional links.

    No product/company names are hardcoded.
    """

    answer_links = get_answer_links(
        scored_links
    )

    best_group = []

    for base_link in answer_links:

        current_group = []

        base_structure = (
            base_link["structure"]
        )

        for other_link in answer_links:

            similarity = (
                structure_similarity(
                    base_structure,
                    other_link[
                        "structure"
                    ]
                )
            )

            if similarity >= 0.40:

                current_group.append(
                    other_link
                )

        if (
            len(current_group)
            >
            len(best_group)
        ):
            best_group = current_group

    # We require multiple links because we're
    # specifically looking for a repeated candidate
    # pattern on the page.
    if len(best_group) < 2:
        return []

    # Remove internal diagnostic field before
    # returning candidates.
    candidates = []

    for link in best_group:

        candidates.append({
            "name":
                clean_candidate_name(
                    link["text"]
                ),

            "text":
                link["text"],

            "evidence":
                link["text"],

            "url":
                link["url"],

            "source_url":
                link["url"],

            "similarity":
                link["similarity"]
        })

    return candidates


# --------------------------------------------------
# Clean candidate name
# --------------------------------------------------

def clean_candidate_name(text):
    """
    Attempts to remove repeated action/value text
    from a candidate card while keeping its name.

    This is generic and does not know any specific
    product names.
    """

    clean_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Cut before common value/action sections.
    # These are interface/action patterns rather
    # than company or product names.
    separators = [
        r"take a closer look",
        r"buy from",
        r"from\s*[$€£¥]",
        r"[$€£¥]\s*[\d,.]+"
    ]

    earliest_position = None

    for separator in separators:

        match = re.search(
            separator,
            clean_text,
            re.IGNORECASE
        )

        if match:

            if (
                earliest_position is None
                or
                match.start()
                < earliest_position
            ):
                earliest_position = (
                    match.start()
                )

    if earliest_position is not None:

        name = clean_text[
            :earliest_position
        ].strip()

        if name:
            return name

    return clean_text


# --------------------------------------------------
# Match clarification against candidates
# --------------------------------------------------

def match_candidate(
    clarification,
    candidates
):
    """
    After the user clarifies, compare that
    clarification ONLY against the candidates
    already discovered.

    No more website crawling occurs.
    """

    if not candidates:

        return {
            "status":
                "not_found"
        }

    clarification_embedding = (
        embedding_model.encode(
            clarification
        )
    )

    candidate_names = [
        candidate["name"]
        for candidate in candidates
    ]

    candidate_embeddings = (
        embedding_model.encode(
            candidate_names
        )
    )

    scores = util.cos_sim(
        clarification_embedding,
        candidate_embeddings
    )[0]

    best_index = (
        scores.argmax().item()
    )

    best_score = (
        scores[best_index].item()
    )

    best_candidate = (
        candidates[best_index]
    )

    print(
        "\nClarification matches:"
    )

    matches = []

    for index, candidate in enumerate(
        candidates
    ):

        matches.append(
            (
                candidate["name"],
                scores[index].item()
            )
        )

    matches.sort(
        key=lambda item:
        item[1],
        reverse=True
    )

    for name, score in matches:

        print(
            f"  {name} "
            f"({score:.3f})"
        )

    if (
            best_score
            < CLARIFICATION_THRESHOLD
    ):
        print(
            "\nClarification did not "
            "match current candidates."
        )

        return {
            "status":
                "no_candidate_match",

            "clarification":
                clarification,

            "similarity":
                best_score
        }

    print(
        "\nSelected candidate:"
    )

    print(
        f"  {best_candidate['name']}"
    )

    return {
        "status":
            "found",

        "candidate":
            best_candidate["name"],

        "evidence":
            best_candidate["evidence"],

        "source_url":
            best_candidate["source_url"],

        "similarity":
            best_score
    }


# --------------------------------------------------
# Main website search
# --------------------------------------------------

def search_website(
    starting_url,
    question_embedding
):

    visited_urls = set()
    pages_to_visit = []

    heapq.heappush(
        pages_to_visit,
        (
            -1.0,
            starting_url
        )
    )

    pages_visited = 0


    while (
        pages_to_visit
        and
        pages_visited < MAX_PAGES
    ):

        negative_score, current_url = (
            heapq.heappop(
                pages_to_visit
            )
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
        # Scrape page
        # --------------------------------

        page = scrape_website(
            current_url
        )

        if page is None:

            print(
                "Scraper could not "
                "retrieve page."
            )

            continue


        # --------------------------------
        # Score ALL links
        # --------------------------------

        scored_links = score_links(
            question_embedding,
            page["links"],
            visited_urls
        )

        scored_links = (
            remove_duplicate_links(
                scored_links
            )
        )


        # --------------------------------
        # Look for candidate group
        # --------------------------------

        answer_links = get_answer_links(
            scored_links
        )

        page_relevance = (
            -negative_score
        )

        if (
            len(answer_links) == 1
            and
            page_relevance >= 0.70
            and
            is_standalone_answer_value(
                answer_links[0]["text"]
            )
        ):

            single_link = answer_links[0]

            candidates = [{
                "name":
                    clean_candidate_name(
                        single_link["text"]
                    ),

                "text":
                    single_link["text"],

                "evidence":
                    single_link["text"],

                "url":
                    single_link["url"],

                "source_url":
                    current_url,

                "similarity":
                    page_relevance
            }]

        else:

            candidates = find_candidate_group(
                scored_links
            )


        # --------------------------------
        # CANDIDATES FOUND -> STOP
        # --------------------------------

        if candidates:

            print(
                "\n=============================="
            )

            print(
                "ANSWER CANDIDATES FOUND"
            )

            print(
                "STOPPING WEBSITE SEARCH"
            )

            print(
                "=============================="
            )

            for candidate in candidates:

                print(
                    f"\n"
                    f"{candidate['name']}"
                )

                print(
                    f"  Evidence: "
                    f"{candidate['evidence']}"
                )

            # STOP. We do not add any more links
            # to the queue and do not crawl again.

            if len(candidates) == 1:

                candidate = candidates[0]

                return {
                    "status":
                        "found",

                    "candidate":
                        candidate["name"],

                    "evidence":
                        candidate["evidence"],

                    "source_url":
                        candidate["source_url"],

                    "similarity":
                        candidate["similarity"]
                }

            return {
                "status":
                    "needs_clarification",

                "reason":
                    "multiple_answer_candidates",

                "candidates":
                    candidates,

                "options":
                    candidates
            }


        # --------------------------------
        # No candidates -> navigate
        # --------------------------------

        relevant_links = [
            link
            for link in scored_links
            if link["similarity"]
            >= LINK_THRESHOLD
        ]

        print(
            "\nNo answer candidates."
        )

        print(
            "Best navigation links:"
        )

        for link in relevant_links[:5]:

            print(
                f"  {link['text']} "
                f"({link['similarity']:.3f})"
            )


        # --------------------------------
        # Add navigation links
        # --------------------------------

        for link in relevant_links:

            if (
                link["url"]
                not in visited_urls
            ):

                heapq.heappush(
                    pages_to_visit,
                    (
                        -link[
                            "similarity"
                        ],

                        link[
                            "url"
                        ]
                    )
                )


    # --------------------------------------------------
    # Nothing found
    # --------------------------------------------------

    print(
        "\nNo answer candidates found."
    )

    return {
        "status":
            "not_found"
    }
