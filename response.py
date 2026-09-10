import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------
# Models
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

GENERATION_MODEL = "google/flan-t5-base"

response_tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL
)

response_model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATION_MODEL
)


# --------------------------------
# Request processing
# --------------------------------

def extract_url(user_input):

    url_pattern = r'https?://[^\s]+'

    match = re.search(
        url_pattern,
        user_input
    )

    if match:
        return match.group(0).rstrip(
            ".,!?"
        )

    return None


def extract_question(
    user_input,
    url
):

    return user_input.replace(
        url,
        ""
    ).strip()


def embed_question(question):

    return embedding_model.encode(
        question
    )


def process_request(user_input):

    url = extract_url(
        user_input
    )

    if not url:
        return None

    question = extract_question(
        user_input,
        url
    )

    question_embedding = embed_question(
        question
    )

    return {
        "url": url,
        "question": question,
        "question_embedding": question_embedding
    }





# --------------------------------
# Text generation
# --------------------------------

def generate_text(
    prompt,
    max_length=150
):

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

    generated_text = (
        response_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    )

    return generated_text.strip()


# --------------------------------
# Grounding helpers
# --------------------------------

def extract_concrete_values(text):

    if not text:
        return []

    patterns = [
        r"[$€£¥]\s*[\d,.]+",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s+[A-Za-z]+"
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        values.extend(
            matches
        )

    return values




# --------------------------------
# Grounded answer builder
# --------------------------------

def build_grounded_fallback(
    question,
    brain_result
):

    candidate = brain_result.get(
        "candidate",
        ""
    ).strip()

    evidence = brain_result.get(
        "evidence",
        ""
    ).strip()

    values = extract_concrete_values(
        evidence
    )

    if not values:

        if evidence:
            return evidence

        return (
            "I couldn't find enough "
            "information to answer that."
        )

    value = values[0]

    question_lower = (
        question.lower()
    )

    evidence_lower = (
        evidence.lower()
    )

    price_question = any(
        phrase in question_lower
        for phrase in [
            "price",
            "cost",
            "how much"
        ]
    )

    if (
            candidate
            and candidate == value
    ):
        return value

    if candidate and price_question:

        if "from" in evidence_lower:

            return (
                f"The {candidate} "
                f"starts at {value}."
            )

        return (
            f"The {candidate} "
            f"costs {value}."
        )

    if candidate:

        return (
            f"{candidate}: {value}"
        )

    return value


# --------------------------------
# Candidate wording
# --------------------------------

def get_shared_candidate_prefix(
    candidates
):

    names = [
        candidate.get(
            "name",
            ""
        ).strip()
        for candidate in candidates
        if candidate.get(
            "name",
            ""
        ).strip()
    ]

    if len(names) < 2:
        return None

    token_lists = [
        re.findall(
            r"[A-Za-z0-9]+",
            name
        )
        for name in names
    ]

    if not token_lists:
        return None

    shortest_length = min(
        len(tokens)
        for tokens in token_lists
    )

    shared_tokens = []

    for index in range(
        shortest_length
    ):

        current_tokens = [
            tokens[index]
            for tokens in token_lists
        ]

        if len({
            token.lower()
            for token in current_tokens
        }) == 1:

            shared_tokens.append(
                current_tokens[0]
            )

        else:
            break

    if not shared_tokens:
        return None

    prefix = " ".join(
        shared_tokens
    )

    if len(prefix) < 2:
        return None

    return prefix


# --------------------------------
# Found response
# --------------------------------

def generate_found_response(
    question,
    brain_result
):

    return build_grounded_fallback(
        question,
        brain_result
    )

# --------------------------------
# Text-based found response
# --------------------------------

def generate_text_found_response(
    question,
    brain_result
):

    evidence = brain_result.get(
        "evidence",
        ""
    ).strip()

    if not evidence:

        return (
            "I couldn't find enough "
            "information to answer that."
        )

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

    answer = generate_text(
        prompt,
        max_length=120
    )

    if not answer:

        return (
            "I couldn't find enough "
            "information to answer that."
        )

    return answer


# --------------------------------
# Clarification response
# --------------------------------

def generate_clarification(
    question,
    brain_result
):
    candidates = brain_result.get(
        "candidates",
        []
    )

    shared_prefix = (
        get_shared_candidate_prefix(
            candidates
        )
    )

    if shared_prefix:

        question_words = {
            word.lower()
            for word in re.findall(
                r"[A-Za-z0-9]+",
                question
            )
        }

        prefix_words = re.findall(
            r"[A-Za-z0-9]+",
            shared_prefix
        )

        supported_words = [
            word
            for word in prefix_words
            if word.lower()
            in question_words
        ]

        if supported_words:

            subject = " ".join(
                supported_words
            )

            return (
                f"Which {subject} option "
                f"are you interested in?"
            )

    return (
        "Which option are you "
        "interested in?"
    )


# --------------------------------
# Not-found response
# --------------------------------

def generate_not_found_response(
    question
):
    return (
        "Sorry, I couldn't find that "
        "information on the website."
    )


# --------------------------------
# Follow-up response
# --------------------------------

def generate_follow_up():
    return (
        "Is there anything else "
        "you'd like help with?"
    )