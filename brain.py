"""
Core search and answer-discovery logic for the website assistant.

This module ranks webpage links using semantic similarity, detects
structured answer candidates, resolves user clarifications, and performs
best-first navigation through websites to locate relevant information.
"""
import heapq
import re

from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer, util
from scraper import scrape_website


# Model used for semantic link scoring and clarification matching.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Search and matching thresholds.
LINK_THRESHOLD = 0.40
MAX_PAGES = 10
CLARIFICATION_THRESHOLD = 0.90


def score_links(question_embedding, links, visited_urls):
    """
    Scores webpage links by semantic similarity to the user's question.

    Filters out empty and previously visited links, embeds the remaining
    link text, and compares each embedding with the question embedding.

    Args:
        question_embedding: Semantic embedding of the user's question.
        links: Links extracted from the current webpage.
        visited_urls: URLs that have already been visited.

    Returns:
        Link dictionaries containing text, URL, and similarity score,
        sorted from highest to lowest similarity.
    """

    # Ignore links without visible text and URLs already searched.
    usable_links = [
        link
        for link in links
        if link["text"].strip() and link["url"] not in visited_urls
    ]

    if not usable_links:
        return []

    # Embed all link labels and compare them with the question at once.
    link_texts = [link["text"] for link in usable_links]
    link_embeddings = embedding_model.encode(link_texts)
    scores = util.cos_sim(question_embedding, link_embeddings)[0]

    scored_links = []

    for index, link in enumerate(usable_links):
        scored_links.append({
            "text": link["text"],
            "url": link["url"],
            "similarity": scores[index].item()
        })

    # Prioritize the most semantically relevant navigation paths.
    scored_links.sort(
        key=lambda link: link["similarity"],
        reverse=True
    )

    return scored_links


def remove_duplicate_links(links):
    """
    Removes duplicate links while preserving their existing order.

    Links are considered duplicates when both their normalized visible
    text and URL are identical.

    Args:
        links: Link dictionaries to filter.

    Returns:
        A list containing the first occurrence of each unique link.
    """

    unique_links = []
    seen = set()

    for link in links:
        # Normalize the text before using the text and URL as a unique key.
        key = (
            link["text"].strip().lower(),
            link["url"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_links.append(link)

    return unique_links


def contains_answer_value(text):
    """
    Checks whether text contains a concrete answer-like value.

    Recognizes currency values, percentages, and numbers followed by a
    unit or word, without relying on specific company or product names.

    Args:
        text: Link text to inspect.

    Returns:
        True if an answer-like value is detected, otherwise False.
    """

    patterns = [
        r"[$€£¥]\s*[\d,.]+",               # Currency
        r"\b\d+(?:\.\d+)?\s*%",            # Percentage
        r"\b\d+(?:\.\d+)?\s+[A-Za-z]+"     # Number followed by unit/word
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def contains_price_value(text):
    """
    Checks whether text contains a concrete currency value.

    Args:
        text: Link text to inspect.

    Returns:
        True if a currency value is detected, otherwise False.
    """

    return bool(re.search(r"[$€£¥]\s*[\d,.]+", text))


def is_standalone_answer_value(text):
    """
    Checks whether the entire text consists of one concrete answer value.

    This allows a strongly relevant page to return a direct value without
    requiring multiple links with a repeated candidate-card structure.

    Args:
        text: Link text to inspect.

    Returns:
        True if the complete text is a currency value, percentage, or
        number followed by a unit or word.
    """

    # Normalize whitespace before requiring the pattern to match all text.
    clean_text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"[$€£¥]\s*[\d,.]+",
        r"\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s+[A-Za-z]+"
    ]

    return any(
        re.fullmatch(pattern, clean_text, re.IGNORECASE)
        for pattern in patterns
    )


def get_link_structure(text):
    """
    Creates a structural fingerprint from answer-bearing link text.

    Replaces changing values with placeholders and keeps the final portion
    of the text so links generated from similar page components can be
    recognized as having the same general structure.

    Args:
        text: Link text to convert into a structural fingerprint.

    Returns:
        A normalized string representing the link's general structure.
    """

    structure = text.lower()

    # Replace specific values so different numbers share the same structure.
    structure = re.sub(
        r"[$€£¥]\s*[\d,.]+(?:\s*[–-]\s*[$€£¥]?\s*[\d,.]+)?",
        " VALUE ",
        structure
    )
    structure = re.sub(r"\d+(?:\.\d+)?", " NUMBER ", structure)

    # Focus on the ending, where repeated action/value text commonly appears.
    words = structure.split()

    if len(words) > 12:
        words = words[-12:]

    structure = " ".join(words)

    # Normalize whitespace created during the replacements.
    structure = re.sub(r"\s+", " ", structure).strip()

    return structure


def structure_similarity(structure_a, structure_b):
    """
    Measures word overlap between two structural fingerprints.

    Uses Jaccard similarity by dividing the number of shared unique words
    by the total number of unique words across both structures.

    Args:
        structure_a: First structural fingerprint.
        structure_b: Second structural fingerprint.

    Returns:
        A similarity score from 0.0 to 1.0.
    """

    words_a = set(structure_a.split())
    words_b = set(structure_b.split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)

    return len(intersection) / len(union)


def get_answer_links(scored_links, price_question=False):
    """
    Filters scored links for links containing concrete answer-like values.

    Price questions specifically require currency evidence so model numbers
    and other numeric text are not incorrectly treated as price answers.

    Args:
        scored_links: Semantically scored links from the current webpage.
        price_question: Whether the user's question is asking for a price.

    Returns:
        Answer-bearing link dictionaries with an added structural fingerprint.
    """

    answer_links = []

    for link in scored_links:
        # Price questions use stricter currency detection than general questions.
        if price_question:
            has_answer_value = contains_price_value(link["text"])
        else:
            has_answer_value = contains_answer_value(link["text"])

        if has_answer_value:
            # Copy the link so its original dictionary is not modified.
            candidate = dict(link)
            candidate["structure"] = get_link_structure(link["text"])
            answer_links.append(candidate)

    return answer_links


def find_candidate_group(scored_links, price_question=False):
    """
    Finds the largest group of structurally similar answer-bearing links.

    Groups links using their structural fingerprints to identify repeated
    answer-card patterns while excluding isolated promotional or unrelated
    links. At least two structurally similar links are required.

    Args:
        scored_links: Semantically scored links from the current webpage.
        price_question: Whether candidate links must contain currency values.

    Returns:
        Candidate dictionaries from the largest repeated structural group,
        or an empty list if no repeated group is found.
    """

    answer_links = get_answer_links(scored_links, price_question)
    best_group = []

    # Treat each answer-bearing link as the base of a possible group.
    for base_link in answer_links:
        current_group = []
        base_structure = base_link["structure"]

        for other_link in answer_links:
            similarity = structure_similarity(
                base_structure,
                other_link["structure"]
            )

            if similarity >= 0.40:
                current_group.append(other_link)

        # Preserve only the largest structural group discovered.
        if len(current_group) > len(best_group):
            best_group = current_group

    # A repeated candidate pattern requires at least two similar links.
    if len(best_group) < 2:
        return []

    candidates = []

    # Return only the information needed by later candidate-matching logic.
    for link in best_group:
        candidates.append({
            "name": clean_candidate_name(link["text"]),
            "evidence": link["text"],
            "url": link["url"],
            "source_url": link["url"],
            "similarity": link["similarity"]
        })

    return candidates


def clean_candidate_name(text):
    """
    Extracts a candidate name from longer answer-card text.

    Normalizes whitespace and removes common action or value sections from
    the end of candidate text without relying on specific product names.

    Args:
        text: The complete candidate-card text.

    Returns:
        The extracted candidate name, or the normalized original text when
        no separator can be used to isolate a name.
    """

    clean_text = re.sub(r"\s+", " ", text).strip()

    # These generic UI/value patterns commonly mark the end of a candidate name.
    separators = [
        r"take a closer look",
        r"buy from",
        r"from\s*[$€£¥]",
        r"[$€£¥]\s*[\d,.]+"
    ]

    earliest_position = None

    # Find whichever separator occurs first in the candidate text.
    for separator in separators:
        match = re.search(separator, clean_text, re.IGNORECASE)

        if match:
            if earliest_position is None or match.start() < earliest_position:
                earliest_position = match.start()

    if earliest_position is not None:
        name = clean_text[:earliest_position].strip()

        if name:
            return name

    return clean_text


def match_candidate(clarification, candidates):
    """
    Matches a user's clarification against previously discovered candidates.

    Embeds the clarification and candidate names, compares them using cosine
    similarity, and selects the strongest match only when it reaches the
    configured clarification threshold. This function does not crawl the
    website again.

    Args:
        clarification: The user's clarification response.
        candidates: Candidate dictionaries discovered by the website search.

    Returns:
        A result dictionary describing a successful candidate match,
        unsuccessful clarification match, or missing candidate set.
    """

    if not candidates:
        return {"status": "not_found"}

    # Compare the clarification semantically with all candidate names.
    clarification_embedding = embedding_model.encode(clarification)
    candidate_names = [candidate["name"] for candidate in candidates]
    candidate_embeddings = embedding_model.encode(candidate_names)

    scores = util.cos_sim(
        clarification_embedding,
        candidate_embeddings
    )[0]

    # Identify the candidate with the highest semantic similarity.
    best_index = scores.argmax().item()
    best_score = scores[best_index].item()
    best_candidate = candidates[best_index]

    print("\nClarification matches:")

    # Sort all candidate scores for readable terminal diagnostics.
    matches = []

    for index, candidate in enumerate(candidates):
        matches.append((
            candidate["name"],
            scores[index].item()
        ))

    matches.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for name, score in matches:
        print(f"  {name} ({score:.3f})")

    # Reject the best candidate if it does not meet the confidence threshold.
    if best_score < CLARIFICATION_THRESHOLD:
        print("\nClarification did not match current candidates.")

        return {
            "status": "no_candidate_match",
            "clarification": clarification,
            "similarity": best_score
        }

    print("\nSelected candidate:")
    print(f"  {best_candidate['name']}")

    return {
        "status": "found",
        "candidate": best_candidate["name"],
        "evidence": best_candidate["evidence"],
        "source_url": best_candidate["source_url"],
        "similarity": best_score
    }


def is_specific_destination(question, navigation_text, url):
    """
    Checks whether a navigated page represents a specific subject in the question.

    Removes generic question and action words, then compares meaningful words
    from the question with words from both the navigation link and URL path.
    At least two shared words are required to identify a specific destination.

    Args:
        question: The user's natural-language question.
        navigation_text: The link text used to reach the current page.
        url: The URL of the current page.

    Returns:
        True if at least two meaningful question words match the destination.
    """

    # Ignore common words that do not help identify the requested subject.
    ignored_words = {
        "what", "which", "who", "where", "when", "why", "how",
        "is", "are", "was", "were", "the", "a", "an", "of", "for", "to",
        "does", "do", "did", "can", "could", "would", "tell", "me", "about",
        "offer", "offers", "offered", "provide", "provides", "provided",
        "price", "cost", "costs", "much"
    }

    question_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", question)
        if word.lower() not in ignored_words
    }

    # Use both link text and URL path because either may identify the destination.
    navigation_words = re.findall(r"[A-Za-z0-9]+", navigation_text)
    path = urlparse(url).path
    path_words = re.findall(r"[A-Za-z0-9]+", path)

    destination_words = {
        word.lower()
        for word in navigation_words + path_words
        if word.lower() not in ignored_words
    }

    matched_words = question_words.intersection(destination_words)

    return len(matched_words) >= 2


def get_dominant_page_currency_value(text):
    """
    Finds a currency value that occurs more often than every other value.

    Counts all currency values in the page text and returns the most frequent
    value only when it has a unique highest count.

    Args:
        text: Visible text extracted from the webpage.

    Returns:
        The uniquely most frequent currency value, or None if no value exists
        or the highest frequency is tied.
    """

    if not text:
        return None

    values = re.findall(r"[$€£¥]\s*[\d,.]+", text)

    if not values:
        return None

    # Count how many times each distinct currency value appears.
    value_counts = {}

    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1

    # Rank currency values from most to least frequent.
    ranked_values = sorted(
        value_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_value, best_count = ranked_values[0]

    if len(ranked_values) == 1:
        return best_value

    second_count = ranked_values[1][1]

    # Only accept the value when it occurs more often than the runner-up.
    if best_count > second_count:
        return best_value

    return None


def search_website(starting_url, question_embedding, question):
    """
    Searches a website for information relevant to the user's question.

    Performs a best-first crawl using semantic link similarity. Each page is
    checked for structured answer candidates before relevant navigation links
    are added to a priority queue. Specific destination pages may also provide
    price or general page-text evidence when structured candidates are absent.

    Args:
        starting_url: The website URL where the search begins.
        question_embedding: Semantic embedding of the user's question.
        question: The user's natural-language question.

    Returns:
        A result dictionary with a status of "found", "found_text",
        "needs_clarification", or "not_found", plus the corresponding answer
        evidence or candidates when available.
    """

    visited_urls = set()
    pages_to_visit = []

    # Start with maximum priority so the initial URL is always visited first.
    heapq.heappush(pages_to_visit, (-1.0, starting_url, ""))

    pages_visited = 0

    # Continue until the priority queue is empty or the page limit is reached.
    while pages_to_visit and pages_visited < MAX_PAGES:
        negative_score, current_url, navigation_text = heapq.heappop(
            pages_to_visit
        )

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        pages_visited += 1

        print(f"\nVisiting: {current_url}")

        page = scrape_website(current_url)

        if page is None:
            print("Scraper could not retrieve page.")
            continue

        # Rank all unvisited links by semantic relevance to the question.
        scored_links = score_links(
            question_embedding,
            page["links"],
            visited_urls
        )
        scored_links = remove_duplicate_links(scored_links)

        # Price questions require currency-bearing candidate evidence.
        price_question = any(
            phrase in question.lower()
            for phrase in ["price", "cost", "how much"]
        )

        answer_links = get_answer_links(scored_links, price_question)

        # Heap priorities are stored as negatives, so reverse the sign here.
        page_relevance = -negative_score

        # A single standalone value can be accepted on a strongly relevant page.
        if (
            len(answer_links) == 1
            and page_relevance >= 0.70
            and is_standalone_answer_value(answer_links[0]["text"])
        ):
            single_link = answer_links[0]

            candidates = [{
                "name": clean_candidate_name(single_link["text"]),
                "text": single_link["text"],
                "evidence": single_link["text"],
                "url": single_link["url"],
                "source_url": current_url,
                "similarity": page_relevance
            }]

        else:
            candidates = find_candidate_group(scored_links, price_question)

        # Stop crawling once structured answer candidates are found.
        if candidates:
            print("\n==============================")
            print("ANSWER CANDIDATES FOUND")
            print("STOPPING WEBSITE SEARCH")
            print("==============================")

            for candidate in candidates:
                print(f"\n{candidate['name']}")
                print(f"  Evidence: {candidate['evidence']}")

            if len(candidates) == 1:
                candidate = candidates[0]

                return {
                    "status": "found",
                    "candidate": candidate["name"],
                    "evidence": candidate["evidence"],
                    "source_url": candidate["source_url"],
                    "similarity": candidate["similarity"]
                }

            return {
                "status": "needs_clarification",
                "candidates": candidates
            }

        # Only sufficiently relevant links are eligible for further navigation.
        relevant_links = [
            link
            for link in scored_links
            if link["similarity"] >= LINK_THRESHOLD
        ]

        # On a specific price destination, fall back to the dominant page price.
        if (
            not candidates
            and not relevant_links
            and navigation_text
            and price_question
            and is_specific_destination(question, navigation_text, current_url)
        ):
            page_value = get_dominant_page_currency_value(page["text"])

            if page_value:
                print("\n==============================")
                print("ANSWER CANDIDATES FOUND")
                print("STOPPING WEBSITE SEARCH")
                print("==============================")

                print(f"\n{page_value}")
                print(f"  Evidence: {page_value}")

                return {
                    "status": "found",
                    "candidate": page_value,
                    "evidence": page_value,
                    "source_url": current_url,
                    "similarity": page_relevance
                }

        # For non-price questions, return page text from a specific destination.
        if (
            not candidates
            and not relevant_links
            and navigation_text
            and not price_question
            and is_specific_destination(question, navigation_text, current_url)
        ):
            page_text = page.get("text", "").strip()

            if page_text:
                print("\n==============================")
                print("TEXT EVIDENCE FOUND")
                print("STOPPING WEBSITE SEARCH")
                print("==============================")

                return {
                    "status": "found_text",
                    "evidence": page_text,
                    "source_url": current_url,
                    "similarity": page_relevance
                }

        print("\nNo answer candidates.")
        print("Best navigation links:")

        for link in relevant_links[:5]:
            print(f"  {link['text']} ({link['similarity']:.3f})")

        # Add relevant unvisited links to the best-first search priority queue.
        for link in relevant_links:
            if link["url"] not in visited_urls:
                heapq.heappush(
                    pages_to_visit,
                    (
                        -link["similarity"],
                        link["url"],
                        link["text"]
                    )
                )

    print("\nNo answer candidates found.")

    return {"status": "not_found"}