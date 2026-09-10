print("MAIN STARTING")

print("Importing response...")

from response import (
    process_request,
    generate_found_response,
    generate_text_found_response,
    generate_clarification,
    generate_not_found_response
)

print("Response imported")

print("Importing brain...")


from brain import (
    search_website,
    match_candidate
)

print("Brain imported")


def main():
    """
    Runs the chatbot in a terminal-based interface.

    This provides a simple way to test the search and
    response pipeline without using the graphical UI.
    """

    user_input = input(
        "What would you like to know? "
    )

    request = process_request(
        user_input
    )

    if request is None:
        print(
            "Please include a valid URL."
        )
        return

    url = request["url"]
    question = request["question"]
    question_embedding = request[
        "question_embedding"
    ]

    # --------------------------------
    # Initial website search
    # --------------------------------

    result = search_website(
        url,
        question_embedding,
        question
    )

    # --------------------------------
    # Conversation loop
    # --------------------------------

    while True:
        status = result["status"]

        # --------------------------------
        # Found structured answer
        # --------------------------------

        if status == "found":
            answer = generate_found_response(
                question,
                result
            )

            print(
                f"\n{answer}"
            )

            break

        # --------------------------------
        # Found page-text answer
        # --------------------------------

        elif status == "found_text":
            answer = generate_text_found_response(
                question,
                result
            )

            print(
                f"\n{answer}"
            )

            break

        # --------------------------------
        # Needs clarification
        # --------------------------------

        elif status == "needs_clarification":
            clarification_question = (
                generate_clarification(
                    question,
                    result
                )
            )

            print(
                f"\n"
                f"{clarification_question}"
            )

            clarification_answer = input(
                "> "
            )

            candidate_result = match_candidate(
                clarification_answer,
                result["candidates"]
            )

            if (
                candidate_result["status"]
                != "no_candidate_match"
            ):
                result = candidate_result
                continue

            # If the clarification does not match the
            # current candidates, refine the original
            # question and continue searching.

            question = (
                f"{question} "
                f"{clarification_answer}"
            )

            refined_request = process_request(
                f"{question} {url}"
            )

            result = search_website(
                url,
                refined_request[
                    "question_embedding"
                ],
                question
            )

        # --------------------------------
        # Not found
        # --------------------------------

        elif status == "not_found":
            answer = (
                generate_not_found_response(
                    question
                )
            )

            print(
                f"\n{answer}"
            )

            break

        # --------------------------------
        # Unexpected search state
        # --------------------------------

        else:
            print(
                "\nSomething went wrong."
            )

            break


if __name__ == "__main__":
    main()