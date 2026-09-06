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
# Clarification context
# --------------------------------

def add_clarification(
    question_context,
    clarification
):

    return (
        question_context.strip()
        + " "
        + clarification.strip()
    )


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


def normalize_value(value):

    return re.sub(
        r"\s+",
        "",
        value
    ).lower()


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

    if not candidates:

        prompt = """
Write one short question asking the user
to provide more specific information.

Output only the question.
"""

        return generate_text(
            prompt,
            max_length=30
        )

    shared_prefix = (
        get_shared_candidate_prefix(
            candidates
        )
    )

    if shared_prefix:

        prompt = f"""
Write one short conversational question.

The user must choose a more specific
option related to:

{shared_prefix}

Ask what specific {shared_prefix}
option they are interested in.

Do not provide an answer.
Do not recommend anything.
Output only the question.
"""

    else:

        prompt = """
Write one short conversational question
asking the user which specific option
they are interested in.

Do not provide an answer.
Do not recommend anything.
Output only the question.
"""

    generated = generate_text(
        prompt,
        max_length=30
    )

    if (
        generated
        and generated.endswith("?")
    ):
        return generated

    if shared_prefix:

        return (
            f"Which {shared_prefix} option "
            f"are you interested in?"
        )

    return (
        "Which specific option "
        "are you interested in?"
    )


# --------------------------------
# Not-found response
# --------------------------------

def generate_not_found_response(
    question
):

    prompt = """
Write one short conversational response.

The requested information could not
be found on the website.

Do not guess.
Do not invent information.
Do not include numbers.
Output only the response.
"""

    generated = generate_text(
        prompt,
        max_length=30
    )

    if (
        generated
        and not extract_concrete_values(
            generated
        )
    ):
        return generated

    return (
        "I couldn't find that "
        "information on the website."
    )


# --------------------------------
# Follow-up response
# --------------------------------

def generate_follow_up():

    prompt = """
Write one short friendly question asking
whether the user would like help with
anything else.

Do not mention any product.
Do not mention any website.
Do not include factual information.
Do not include numbers.

Output only the question.
"""

    generated = generate_text(
        prompt,
        max_length=25
    )

    if (
        generated
        and generated.endswith("?")
        and not extract_concrete_values(
            generated
        )
    ):
        return generated

    return (
        "Is there anything else "
        "you'd like help with?"
    )