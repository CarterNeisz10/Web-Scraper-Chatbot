"""
Terminal interface for the website assistant.

This module provides a simple command-line interface for testing the
website search, answer generation, and clarification pipeline without
using the graphical user interface.
"""
from response import (
    process_request,
    generate_found_response,
    generate_text_found_response,
    generate_clarification,
    generate_not_found_response
)

from brain import search_website, match_candidate


def main():
    """
    Runs the website assistant through a terminal interface.

    Processes a natural-language question containing a URL, searches the
    website for relevant information, and displays either a structured
    answer, a text-based answer, or a clarification request. If a
    clarification does not match the existing candidates, the question is
    refined and the website search continues.
    """

    user_input = input("What would you like to know? ")
    request = process_request(user_input)

    if request is None:
        print("Please include a valid URL.")
        return

    url = request["url"]
    question = request["question"]
    question_embedding = request["question_embedding"]

    # Perform the initial search using the user's question and its embedding.
    result = search_website(url, question_embedding, question)

    # Continue until the search produces a final answer or fails.
    while True:
        status = result["status"]

        if status == "found":
            answer = generate_found_response(question, result)
            print(f"\n{answer}")
            break

        elif status == "found_text":
            answer = generate_text_found_response(question, result)
            print(f"\n{answer}")
            break

        elif status == "needs_clarification":
            clarification_question = generate_clarification(question, result)
            print(f"\n{clarification_question}")

            clarification_answer = input("> ")
            candidate_result = match_candidate(
                clarification_answer,
                result["candidates"]
            )

            # A successful match resolves the ambiguity without another search.
            if candidate_result["status"] != "no_candidate_match":
                result = candidate_result
                continue

            # Otherwise, add the clarification to the question and search again.
            question = f"{question} {clarification_answer}"
            refined_request = process_request(f"{question} {url}")

            result = search_website(
                url,
                refined_request["question_embedding"],
                question
            )

        elif status == "not_found":
            answer = generate_not_found_response(question)
            print(f"\n{answer}")
            break

        else:
            print("\nSomething went wrong.")
            break


if __name__ == "__main__":
    main()