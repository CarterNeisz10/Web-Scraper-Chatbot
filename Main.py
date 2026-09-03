from response import (
    process_request,
    embed_question,
    generate_found_response,
    generate_clarification,
    generate_not_found_response,
    add_clarification
)

from brain import search_website


def main():

    # --------------------------------
    # Get original request
    # --------------------------------

    user_input = input(
        "What would you like to know? "
    )

    request = process_request(user_input)

    if request is None:
        print("Please include a valid URL.")
        return

    url = request["url"]
    question = request["question"]
    question_embedding = request[
        "question_embedding"
    ]

    # --------------------------------
    # Conversation/search loop
    # --------------------------------

    while True:

        result = search_website(
            url,
            question_embedding
        )

        status = result["status"]

        # --------------------------------
        # FOUND
        # --------------------------------

        if status == "found":

            answer = generate_found_response(
                question,
                result
            )

            print(f"\n{answer}")
            break

        # --------------------------------
        # NEEDS CLARIFICATION
        # --------------------------------

        elif status == "needs_clarification":

            clarification_question = (
                generate_clarification(
                    question,
                    result
                )
            )

            print(
                f"\n{clarification_question}"
            )

            clarification_answer = input("> ")

            # Rewrite the question using the
            # user's new information
            question = add_clarification(
                question,
                clarification_answer
            )

            print(
                f"\nUpdated question: {question}"
            )

            # Create a NEW embedding from the
            # more specific question
            question_embedding = embed_question(
                question
            )

            # Loop runs brain again

        # --------------------------------
        # NOT FOUND
        # --------------------------------

        elif status == "not_found":

            answer = generate_not_found_response(
                question
            )

            print(f"\n{answer}")
            break

        # --------------------------------
        # Unexpected result
        # --------------------------------

        else:
            print(
                "\nSomething went wrong while "
                "processing the request."
            )
            break


if __name__ == "__main__":
    main()