print("MAIN STARTING")

print("Importing response...")

from response import (
    process_request,
    generate_found_response,
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
        question_embedding
    )


    # --------------------------------
    # Conversation loop
    # --------------------------------

    while True:

        status = result["status"]


        # --------------------------------
        # FOUND
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
        # NEEDS CLARIFICATION
        # --------------------------------

        elif (
            status
            ==
            "needs_clarification"
        ):

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


            # IMPORTANT:
            #
            # We do NOT search the website again.
            #
            # We only compare the clarification
            # against candidates already found.

            result = match_candidate(
                clarification_answer,
                result["candidates"]
            )


        # --------------------------------
        # NOT FOUND
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
        # Unexpected
        # --------------------------------

        else:

            print(
                "\nSomething went wrong."
            )

            break


if __name__ == "__main__":
    main()