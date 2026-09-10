import tkinter as tk

from response import (
    process_request,
    generate_found_response,
    generate_text_found_response,
    generate_clarification,
    generate_not_found_response,
    generate_follow_up
)

from brain import (
    search_website,
    match_candidate
)


# --------------------------------
# Conversation state
# --------------------------------

current_question = None
current_candidates = None
current_url = None


# --------------------------------
# Reset conversation
# --------------------------------

def reset_conversation():

    global current_question
    global current_candidates
    global current_url

    current_question = None
    current_candidates = None
    current_url = None


# --------------------------------
# Display assistant response
# --------------------------------

def add_assistant_message(message):

    if message:
        add_message(
            f"Assistant: {message}"
        )


# --------------------------------
# Complete current request
# --------------------------------

def finish_request(answer):

    add_assistant_message(
        answer
    )

    follow_up = generate_follow_up()

    add_assistant_message(
        follow_up
    )

    reset_conversation()


# --------------------------------
# Send message
# --------------------------------

def send_message():


    global current_question
    global current_candidates
    global current_url



    user_text = input_box.get().strip()

    if not user_text:
        return

    input_box.delete(
        0,
        tk.END
    )

    add_message(
        f"You: {user_text}"
    )


    # --------------------------------
    # Existing clarification
    # --------------------------------

    if current_candidates is not None:

        result = match_candidate(
            user_text,
            current_candidates
        )

        if result["status"] == "found":

            answer = generate_found_response(
                current_question,
                result
            )

            finish_request(
                answer
            )


        elif (

                result["status"]

                == "no_candidate_match"

        ):

            new_question = (

                    current_question

                    + " "

                    + user_text

            )

            new_request = process_request(

                new_question

                + " "

                + current_url

            )

            current_question = new_question

            current_candidates = None

            result = search_website(

                current_url,

                new_request[
                    "question_embedding"
                ],

                new_question

            )

            if result["status"] == "found":

                answer = generate_found_response(

                    current_question,

                    result

                )

                finish_request(

                    answer

                )


            elif result["status"] == "found_text":

                answer = generate_text_found_response(

                    current_question,

                    result

                )

                finish_request(

                    answer

                )


            elif (

                    result["status"]

                    == "needs_clarification"

            ):

                current_candidates = result[

                    "candidates"

                ]

                clarification = generate_clarification(

                    current_question,

                    result

                )

                add_assistant_message(

                    clarification

                )


            else:

                answer = generate_not_found_response(

                    current_question

                )

                finish_request(

                    answer

                )

        return


    # --------------------------------
    # New request
    # --------------------------------

    request = process_request(
        user_text
    )

    if request is None:

        answer = generate_not_found_response(
            user_text
        )

        add_assistant_message(
            answer
        )

        return


    current_question = request[
        "question"
    ]

    current_url = request[
        "url"
    ]

    result = search_website(
        request["url"],
        request["question_embedding"],
        current_question
    )


    # --------------------------------
    # Found
    # --------------------------------

    if result["status"] == "found":

        answer = generate_found_response(
            current_question,
            result
        )

        finish_request(
            answer
        )


    # --------------------------------
    # Text answer found
    # --------------------------------

    elif result["status"] == "found_text":

        answer = generate_text_found_response(
            current_question,
            result
        )

        finish_request(
            answer
        )


    # --------------------------------
    # Needs clarification
    # --------------------------------

    elif (
        result["status"]
        == "needs_clarification"
    ):

        current_candidates = result[
            "candidates"
        ]

        clarification = generate_clarification(
            current_question,
            result
        )

        add_assistant_message(
            clarification
        )


    # --------------------------------
    # Not found
    # --------------------------------

    elif result["status"] == "not_found":

        answer = generate_not_found_response(
            current_question
        )

        finish_request(
            answer
        )


# --------------------------------
# Add message to chat
# --------------------------------

def add_message(message):

    chat_box.config(
        state="normal"
    )

    chat_box.insert(
        tk.END,
        message + "\n\n"
    )

    chat_box.config(
        state="disabled"
    )

    chat_box.see(
        tk.END
    )


# --------------------------------
# Window
# --------------------------------

window = tk.Tk()

window.title(
    "Website Assistant"
)

window.geometry(
    "600x500"
)


# --------------------------------
# Title
# --------------------------------

title = tk.Label(
    window,
    text="Website Assistant",
    font=("Arial", 20, "bold")
)

title.pack(
    pady=(20, 5)
)


subtitle = tk.Label(
    window,
    text="Ask a question about any website"
)

subtitle.pack(
    pady=(0, 15)
)


# --------------------------------
# Chat area
# --------------------------------

chat_box = tk.Text(
    window,
    height=17,
    width=65,
    wrap="word",
    state="disabled"
)

chat_box.pack(
    padx=20,
    pady=10
)


# --------------------------------
# Input
# --------------------------------

input_box = tk.Entry(
    window,
    width=55
)

input_box.pack(
    padx=20,
    pady=(10, 5)
)


# --------------------------------
# Send button
# --------------------------------

send_button = tk.Button(
    window,
    text="Send",
    command=send_message
)

send_button.pack(
    pady=10
)


# -------------------------------
# Enter key
# --------------------------------

window.bind(
    "<Return>",
    lambda event: send_message()
)


# --------------------------------
# Start UI
# --------------------------------

window.mainloop()