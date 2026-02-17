"""Prompt formatting helpers for chat templates."""

from __future__ import annotations

from cortex.conversation_manager import MessageRole


def format_prompt_with_chat_template(
    *,
    conversation_manager,
    model_manager,
    template_registry,
    user_input: str,
    include_user: bool = True,
    logger=None,
) -> str:
    """Format the prompt with the appropriate chat template for the model."""
    # Get current conversation context
    conversation = conversation_manager.get_current_conversation()

    # Get the tokenizer for the current model
    model_name = model_manager.current_model
    tokenizer = model_manager.tokenizers.get(model_name)

    # Build messages list from conversation history
    messages = []
    if conversation and conversation.messages:
        for msg in conversation.messages[-20:]:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

    # Add current user message
    if include_user:
        messages.append({
            "role": "user",
            "content": user_input
        })

    # Use template registry to format messages
    try:
        profile = template_registry.setup_model(
            model_name,
            tokenizer=tokenizer,
            interactive=False
        )
        formatted = profile.format_messages(messages, add_generation_prompt=True)
        return formatted if isinstance(formatted, str) else str(formatted)

    except (AttributeError, TypeError, ValueError) as e:
        if logger is not None:
            logger.debug(f"Template registry failed: {e}, using fallback")

        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted if isinstance(formatted, str) else str(formatted)
            except (AttributeError, TypeError, ValueError) as e:
                if logger is not None:
                    logger.debug(f"Tokenizer apply_chat_template failed: {e}")

    # Fallback: For TinyLlama and other chat models, use the proper format
    if model_name and "chat" in model_name.lower():
        history = ""
        if conversation and conversation.messages:
            recent_messages = conversation.messages[-6:]
            for msg in recent_messages:
                if msg.role == MessageRole.USER:
                    history += f"<|user|>\n{msg.content}</s>\n"
                elif msg.role == MessageRole.ASSISTANT:
                    history += f"<|assistant|>\n{msg.content}</s>\n"

        prompt = f"{history}<|user|>\n{user_input}</s>\n<|assistant|>\n"
        return prompt

    # Generic fallback for non-chat models
    if conversation and len(conversation.messages) > 0:
        context = ""
        recent_messages = conversation.messages[-6:]
        for msg in recent_messages:
            if msg.role == MessageRole.USER:
                context += f"User: {msg.content}\n"
            elif msg.role == MessageRole.ASSISTANT:
                context += f"Assistant: {msg.content}\n"

        prompt = f"{context}User: {user_input}\nAssistant:"
    else:
        prompt = f"User: {user_input}\nAssistant:"

    return prompt
