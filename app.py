"""
Minimal Flask application for a LINE Messaging API bot on Render.

This application defines two routes:
 - `GET /` returns a simple message letting users know the service is running.
 - `POST /line-webhook` handles incoming webhook events from LINE.

The LINE channel secret and access token are expected to be set in the
environment variables `LINE_CHANNEL_SECRET` and `LINE_CHANNEL_ACCESS_TOKEN`.
When a text message is received, the bot echoes the same text back to the user.

This file can be deployed on Render with a `render.yaml` specifying
`gunicorn app:app --bind 0.0.0.0:$PORT` as the start command.
"""

import os
from flask import Flask, request, abort

try:
    # Import the LINE SDK components. These may not be installed locally,
    # so users should add `line-bot-sdk` to their requirements.
    from linebot import LineBotApi, WebhookParser
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
except ImportError:
    # Provide a clear error if dependencies are missing.
    raise ImportError(
        "Missing dependencies. Please install the `line-bot-sdk` package via pip."
    )


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load LINE credentials from environment variables. Render lets you
    # configure environment variables in the service settings.
    channel_secret = os.getenv("LINE_CHANNEL_SECRET")
    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not channel_secret or not channel_access_token:
        raise RuntimeError(
            "Missing LINE credentials. Set LINE_CHANNEL_SECRET and "
            "LINE_CHANNEL_ACCESS_TOKEN in your environment variables."
        )

    # Initialize the LINE API client and parser.
    line_bot_api = LineBotApi(channel_access_token)
    parser = WebhookParser(channel_secret)

    @app.route("/", methods=["GET"])
    def index():
        """Return a simple greeting message for the root path."""
        return (
            "LINE Bot service is running. This endpoint is not meant to be used "
            "for messaging; it simply confirms that the server is up.",
            200,
        )

    @app.route("/line-webhook", methods=["POST"])
    def callback():
        """Handle webhook events from the LINE platform."""
        # Retrieve the signature from the `X-Line-Signature` header.
        signature = request.headers.get("X-Line-Signature", "")

        # Get the request body as text (LINE sends JSON strings).
        body = request.get_data(as_text=True)

        # Parse and validate the request. If the signature is invalid,
        # `InvalidSignatureError` will be raised and we abort with 400.
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            abort(400, description="Invalid signature")

        # Handle the events returned by the parser. LINE may batch multiple
        # webhook events into a single POST request.
        for event in events:
            # Only handle text messages. Other event types (images, stickers,
            # follow events, etc.) can be handled here as needed.
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
                # Echo back the received text.
                try:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text=event.message.text),
                    )
                except Exception:
                    # In production, you should log the exception and handle
                    # errors gracefully. Here we simply ignore failures.
                    pass
        # Return 200 OK so that LINE knows the webhook was processed successfully.
        return "OK", 200

    return app


# Create a module-level application object for gunicorn to import.
app = create_app()
