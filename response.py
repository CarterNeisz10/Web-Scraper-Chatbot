"""
Request processing and response generation for the website assistant.

This module extracts questions and URLs from user input, creates semantic
question embeddings, generates grounded answers from website evidence,
and produces clarification, not-found, and follow-up responses.
"""
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load the embedding model used to represent questions semantically.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the text-generation model used for grounded page-text responses.
GENERATION_MODEL = "google/flan-t5-base"

response_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
response_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)


def extract_url(user_input):
    """
    Extracts the first HTTP or HTTPS URL from a user's input.

    Args:
        user_input: The complete text entered by the user.

    Returns:
        The extracted URL with trailing punctuation removed, or None if
        no valid HTTP or HTTPS URL is found.
    """

    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, user_input)

    if match:
        return match.group(0).rstrip(".,!?")

    return None


def extract_question(user_input, url):
    """
    Separates the natural-language question from its URL.

    Args:
        user_input: The complete text entered by the user.
        url: The URL previously extracted from the input.

    Returns:
        The remaining question text with surrounding whitespace removed.
    """

    return user_input.replace(url, "").strip()


def embed_question(question):
    """
    Converts a natural-language question into a semantic embedding.

    Args:
        question: The question to encode.

    Returns:
        The numerical embedding produced by the sentence-transformer model.
    """

    return embedding_model.encode(question)


def process_request(user_input):
    """
    Converts raw user input into the components required by the search system.

    Extracts the website URL and question, then creates a semantic embedding
    of the question for similarity-based website navigation.

    Args:
        user_input: The complete text entered by the user.

    Returns:
        A dictionary containing the URL, question, and question embedding.
        Returns None if no URL is found.
    """

    url = extract_url(user_input)

    if not url:
        return None

    question = extract_question(user_input, url)
    question_embedding = embed_question(question)

    return {
        "url": url,
        "question": question,
        "question_embedding": question_embedding
    }


def generate_text(prompt, max_length=150):
    """
    Generates deterministic text from a prompt using the response model.

    Args:
        prompt: The text prompt supplied to the generation model.
        max_length: The maximum number of new tokens the model may generate.

    Returns:
        The generated text with surrounding whitespace removed.
    """

    # Convert the prompt into tensors that the generation model can process.
    inputs = response_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = response_model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False
    )

    generated_text = response_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated_text.strip()


def extract_concrete_values(text):
    """
    Extracts common numerical values from evidence text.

    Detects currency amounts, percentages, and numbers followed by a word
    so structured answers can remain grounded in scraped evidence.

    Args:
        text: The evidence text to search.

    Returns:
        A list of concrete values found in the text.
    """

    if not text:
        return []

    patterns = [
        r"[$€£¥]\s*[\d,.]+",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s+[A-Za-z]+"
    ]

    values = []

    # Run each value pattern independently and combine all matches.
    for pattern in patterns:
        matches = re.findall(pattern, text)
        values.extend(matches)

    return values


def build_grounded_fallback(question, brain_result):
    """
    Builds a structured answer using only values found in website evidence.

    Uses the candidate and evidence returned by the search system to produce
    concise responses for concrete values such as prices. If no concrete
    value exists, the original evidence is returned instead.

    Args:
        question: The user's natural-language question.
        brain_result: The result dictionary returned by the search system.

    Returns:
        A grounded response constructed from the available evidence.
    """

    candidate = brain_result.get("candidate", "").strip()
    evidence = brain_result.get("evidence", "").strip()

    values = extract_concrete_values(evidence)

    # Fall back to the evidence itself when no concrete value was extracted.
    if not values:
        if evidence:
            return evidence

        return "I couldn't find enough information to answer that."

    value = values[0]

    question_lower = question.lower()
    evidence_lower = evidence.lower()

    price_question = any(
        phrase in question_lower
        for phrase in ["price", "cost", "how much"]
    )

    # A standalone value, such as a scraped price, needs no additional wording.
    if candidate and candidate == value:
        return value

    # Add natural wording when a named candidate is associated with a price.
    if candidate and price_question:
        if "from" in evidence_lower:
            return f"The {candidate} starts at {value}."

        return f"The {candidate} costs {value}."

    if candidate:
        return f"{candidate}: {value}"

    return value


def get_shared_candidate_prefix(candidates):
    """
    Finds the shared leading words among multiple candidate names.

    Args:
        candidates: Candidate dictionaries containing a name field.

    Returns:
        The shared candidate prefix, or None if no useful prefix exists.
    """

    # Ignore candidates that do not contain a usable name.
    names = [
        candidate.get("name", "").strip()
        for candidate in candidates
        if candidate.get("name", "").strip()
    ]

    if len(names) < 2:
        return None

    # Split each candidate name into comparable alphanumeric tokens.
    token_lists = [
        re.findall(r"[A-Za-z0-9]+", name)
        for name in names
    ]

    if not token_lists:
        return None

    shortest_length = min(len(tokens) for tokens in token_lists)
    shared_tokens = []

    # Compare the candidates word-by-word until their prefixes diverge.
    for index in range(shortest_length):
        current_tokens = [tokens[index] for tokens in token_lists]

        if len({token.lower() for token in current_tokens}) == 1:
            shared_tokens.append(current_tokens[0])
        else:
            break

    if not shared_tokens:
        return None

    prefix = " ".join(shared_tokens)

    if len(prefix) < 2:
        return None

    return prefix


def generate_found_response(question, brain_result):
    """
    Generates a response for a structured search result.

    Args:
        question: The user's natural-language question.
        brain_result: The structured result returned by the search system.

    Returns:
        A grounded response built from the result's evidence.
    """

    return build_grounded_fallback(question, brain_result)


def generate_text_found_response(question, brain_result):
    """
    Generates an answer from unstructured webpage text.

    Provides scraped page evidence to the generation model and instructs it
    to answer using only that evidence rather than outside knowledge.

    Args:
        question: The user's natural-language question.
        brain_result: The search result containing webpage evidence.

    Returns:
        A generated evidence-grounded answer, or a fallback message when
        usable evidence or generated text is unavailable.
    """

    evidence = brain_result.get("evidence", "").strip()

    if not evidence:
        return "I couldn't find enough information to answer that."

    # Explicitly restrict generation to information contained in the page.
    prompt = (
        "Answer the question using only "
        "the website evidence below. "
        "Do not use outside knowledge. "
        "If the evidence does not contain "
        "the answer, say that the answer "
        "was not found in the evidence.\n\n"
        f"Question: {question}\n\n"
        f"Website evidence: {evidence}\n\n"
        "Answer:"
    )

    answer = generate_text(prompt, max_length=120)

    if not answer:
        return "I couldn't find enough information to answer that."

    return answer


def generate_clarification(question, brain_result):
    """
    Creates a clarification question for ambiguous search results.

    Attempts to identify a shared candidate prefix that is also supported
    by the user's original question. This produces a more specific prompt
    without introducing terminology that the user did not provide.

    Args:
        question: The user's original natural-language question.
        brain_result: The search result containing multiple candidates.

    Returns:
        A specific clarification question when possible, otherwise a
        generic request to choose an option.
    """

    candidates = brain_result.get("candidates", [])
    shared_prefix = get_shared_candidate_prefix(candidates)

    if shared_prefix:
        # Only use shared prefix words that also occur in the user's question.
        question_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", question)
        }

        prefix_words = re.findall(r"[A-Za-z0-9]+", shared_prefix)

        supported_words = [
            word
            for word in prefix_words
            if word.lower() in question_words
        ]

        if supported_words:
            subject = " ".join(supported_words)
            return f"Which {subject} option are you interested in?"

    return "Which option are you interested in?"


def generate_not_found_response(question):
    """
    Returns the standard response used when website information is not found.

    Args:
        question: The user's question.

    Returns:
        The standard not-found message.
    """

    return "Sorry, I couldn't find that information on the website."


def generate_follow_up():
    """
    Returns the standard follow-up message after a completed request.

    Returns:
        A message inviting the user to ask another question.
    """

    return "Is there anything else you'd like help with?"