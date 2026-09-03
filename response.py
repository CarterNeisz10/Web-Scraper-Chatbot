import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Models
# --------------------------------------------------

# Used ONLY for creating semantic vectors
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Used ONLY for generating user-facing language
GENERATION_MODEL = "google/flan-t5-base"

response_tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL
)

response_model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATION_MODEL
)


# --------------------------------------------------
# User request processing
# --------------------------------------------------

def extract_url(user_input):
    """
    Extracts the URL from the user's input.
    """
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, user_input)

    if match:
        return match.group(0).rstrip(".,!?")

    return None


def extract_question(user_input, url):
    """
    Removes the URL before any embedding occurs.
    """
    return user_input.replace(url, "").strip()


def embed_question(question):
    """
    Converts user-provided question context into
    a semantic vector.

    No generative model is involved.
    """
    return embedding_model.encode(question)


def process_request(user_input):
    """
    Extracts the website URL and embeds ONLY
    the user's question.
    """
    url = extract_url(user_input)

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


def add_clarification(
    question_context,
    clarification
):
    """
    Adds the user's clarification directly to the
    existing question context.

    IMPORTANT:
    No generative AI rewrites user input here.
    We preserve the user's actual words.
    """
    return (
        question_context.strip()
        + " "
        + clarification.strip()
    )


# --------------------------------------------------
# Generative model
# --------------------------------------------------

def generate_text(prompt, max_length=150):
    """
    Generates USER-FACING language.

    This model is never used to determine the
    semantic meaning sent to brain.py.
    """
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


# --------------------------------------------------
# FOUND
# --------------------------------------------------

def generate_found_response(
    question,
    brain_result
):
    """
    Generates the final user-facing answer from
    evidence found by brain.py.
    """
    evidence = brain_result["evidence"]

    prompt = f"""
Answer the user's question using only the website evidence.

User question:
{question}

Website evidence:
{evidence}

Give a direct, natural answer.
Do not mention similarity scores.
Do not mention the brain or scraper.
Do not invent information.
"""

    return generate_text(prompt)


# --------------------------------------------------
# NEEDS CLARIFICATION
# --------------------------------------------------

def generate_clarification(
    question,
    brain_result
):
    """
    Generates a natural clarification question.

    The model should determine WHAT information is
    missing rather than listing raw website evidence.
    """
    reason = brain_result.get(
        "reason",
        "The request is ambiguous."
    )

    prompt = f"""
The user asked:

{question}

The website search determined that the request is ambiguous.

Reason:
{reason}

Ask ONE short and natural question that would clarify
what the user means.

For example, if they ask for the price of a product but
there are several versions of that product, ask which
version they mean.

Do not answer the original question.
Do not list website evidence.
Do not make up product names.
Keep the question short.
"""

    return generate_text(
        prompt,
        max_length=50
    )


# --------------------------------------------------
# NOT FOUND
# --------------------------------------------------

def generate_not_found_response(question):
    """
    Generates a natural response when brain.py
    cannot find sufficient evidence.
    """
    prompt = f"""
The user asked:

{question}

The website search could not find sufficient reliable
information to answer the question.

Respond naturally and briefly.
Do not invent an answer.
"""

    return generate_text(
        prompt,
        max_length=60
    )