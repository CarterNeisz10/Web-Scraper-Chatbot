"""
Graphical user interface for the website assistant.

This module provides the Tkinter-based chat interface and manages
conversation state, user input, clarification requests, and the display
of responses returned by the search and response pipeline.
"""
import tkinter as tk

from response import (
    process_request,
    generate_found_response,
    generate_text_found_response,
    generate_clarification,
    generate_not_found_response,
    generate_follow_up
)

from brain import search_website, match_candidate


# Stores an active question when the assistant is waiting for clarification.
current_question = None
current_candidates = None
current_url = None


def reset_conversation():
    """
    Clears the stored conversation state after a request is completed.

    Resets the current question, candidate list, and website URL so the
    next user message is processed as a new request.
    """

    global current_question
    global current_candidates
    global current_url

    current_question = None
    current_candidates = None
    current_url = None


def add_assistant_message(message):
    """
    Displays a message from the assistant in the chat window.

    Args:
        message: The text to display. Empty messages are ignored.
    """

    if message:
        add_message(f"Assistant: {message}")


def finish_request(answer):
    """
    Completes a request and resets the conversation state.

    Displays the final answer, adds a follow-up message, and clears any
    stored question, candidate, and URL information.

    Args:
        answer: The final answer to display to the user.
    """

    add_assistant_message(answer)

    follow_up = generate_follow_up()
    add_assistant_message(follow_up)

    reset_conversation()


def send_message():
    """
    Processes a message entered through the graphical interface.

    Handles both new questions and clarification responses. New questions
    are converted into embeddings and passed to the website search.
    Clarifications are first compared with existing candidates; if no
    candidate matches, the clarification is added to the original question
    and the website is searched again.
    """

    global current_question
    global current_candidates
    global current_url

    user_text = input_box.get().strip()

    if not user_text:
        return

    # Clear the input field and display the submitted message.
    input_box.delete(0, tk.END)
    add_message(f"You: {user_text}")

    # If candidates are stored, this message is treated as a clarification.
    if current_candidates is not None:
        result = match_candidate(user_text, current_candidates)

        if result["status"] == "found":
            answer = generate_found_response(current_question, result)
            finish_request(answer)

        elif result["status"] == "no_candidate_match":
            # Refine the original question when the clarification does not
            # match any of the candidates already discovered.
            new_question = current_question + " " + user_text
            new_request = process_request(new_question + " " + current_url)

            current_question = new_question
            current_candidates = None

            result = search_website(
                current_url,
                new_request["question_embedding"],
                new_question
            )

            if result["status"] == "found":
                answer = generate_found_response(current_question, result)
                finish_request(answer)

            elif result["status"] == "found_text":
                answer = generate_text_found_response(current_question, result)
                finish_request(answer)

            elif result["status"] == "needs_clarification":
                # Preserve the new candidates so the next message can
                # attempt to resolve the remaining ambiguity.
                current_candidates = result["candidates"]
                clarification = generate_clarification(
                    current_question,
                    result
                )
                add_assistant_message(clarification)

            else:
                answer = generate_not_found_response(current_question)
                finish_request(answer)

        return

    # No candidates are stored, so process this as a new website question.
    request = process_request(user_text)

    if request is None:
        answer = generate_not_found_response(user_text)
        add_assistant_message(answer)
        return

    current_question = request["question"]
    current_url = request["url"]

    result = search_website(
        current_url,
        request["question_embedding"],
        current_question
    )

    if result["status"] == "found":
        answer = generate_found_response(current_question, result)
        finish_request(answer)

    elif result["status"] == "found_text":
        answer = generate_text_found_response(current_question, result)
        finish_request(answer)

    elif result["status"] == "needs_clarification":
        # Keep the candidates in conversation state for the next message.
        current_candidates = result["candidates"]

        clarification = generate_clarification(current_question, result)
        add_assistant_message(clarification)

    elif result["status"] == "not_found":
        answer = generate_not_found_response(current_question)
        finish_request(answer)


def add_message(message):
    """
    Adds a message to the Tkinter chat display.

    Temporarily enables the read-only text widget, inserts the message,
    disables editing again, and scrolls to the newest content.

    Args:
        message: The formatted message to display.
    """

    chat_box.config(state="normal")
    chat_box.insert(tk.END, message + "\n\n")
    chat_box.config(state="disabled")
    chat_box.see(tk.END)


# Create the main application window.
window = tk.Tk()
window.title("Website Assistant")
window.geometry("600x500")


# Create the heading and description.
title = tk.Label(
    window,
    text="Website Assistant",
    font=("Arial", 20, "bold")
)
title.pack(pady=(20, 5))

subtitle = tk.Label(
    window,
    text="Ask a question about any website"
)
subtitle.pack(pady=(0, 15))


# Create the read-only conversation display.
chat_box = tk.Text(
    window,
    height=17,
    width=65,
    wrap="word",
    state="disabled"
)
chat_box.pack(padx=20, pady=10)


# Create the user input field.
input_box = tk.Entry(window, width=55)
input_box.pack(padx=20, pady=(10, 5))


# Allow messages to be submitted by button or Enter key.
send_button = tk.Button(
    window,
    text="Send",
    command=send_message
)
send_button.pack(pady=10)

window.bind("<Return>", lambda event: send_message())


# Start Tkinter's event loop.
window.mainloop()