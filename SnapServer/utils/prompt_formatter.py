"""
Prompt Formatter
================
Converts OpenAI-style messages into a prompt string for local models.
This keeps the formatter completely model-agnostic.
"""


import datetime
import logging
from typing import List, Dict, Optional, Any
logger = logging.getLogger(__name__)


class PromptFormatter:
    """
    Model-agnostic prompt formatter. No hardcoded templates.

    Usage:
        formatter = PromptFormatter()

        # With HF tokenizer (best — uses model's own template)
        prompt = formatter.format(messages, tokenizer=tokenizer)

        # With Jinja2 template string (from tokenizer_config.json or models.yaml)
        prompt = formatter.format(messages, template=jinja2_string)

        # Generic fallback (Role: Content format)
        prompt = formatter.format(messages)
    """

    DEFAULT_SYSTEM = "You are a helpful AI assistant."

    def format(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Optional[Any] = None,
        template: Optional[str] = None,
        default_system: Optional[str] = None,
        add_generation_prompt: bool = True,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """
        Format messages into a prompt string.

        Args:
            messages:               List of {"role": ..., "content": ...} dicts.
            tokenizer:              HuggingFace tokenizer with apply_chat_template().
            template:               Jinja2 chat template string.
                                    Source: tokenizer_config.json (auto-detected by genie_plugin.py) or models.yaml chat_template field.
            default_system:         System prompt to inject if none present.
            add_generation_prompt:  Append the assistant turn opener.
            tools:                  Optional list of OpenAI-format tool dicts.  When provided the HF tokenizer path passes them directly to apply_chat_template(); all other paths inject a JSON tool-schemas block into the system message.

        Returns:
            Formatted prompt string.
        """
        sys_prompt = default_system or self.DEFAULT_SYSTEM

        # Ensure there is a system message
        msgs = list(messages)
        if not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": sys_prompt}] + msgs

        # HuggingFace tokenizer
        # Uses the model's own apply_chat_template() — most accurate.
        # Requires: pip install transformers
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                kwargs: Dict[str, Any] = dict(
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                if tools:
                    kwargs["tools"] = tools
                return tokenizer.apply_chat_template(msgs, **kwargs)
            except Exception:
                pass  # fall through

        # Jinja2 template string : pass tools directly into the render context so templates with a native `{% if tools %}` block (most HF tool-calling templates) can use them, same as the HF-tokenizer path above.
        # template is sourced from tokenizer_config.json (auto-detected, no hardcoding) or from models.yaml chat_template field (user override).
        # Requires: pip install jinja2
        if template:
            try:
                return self._render_jinja(msgs, template, add_generation_prompt, tools=tools)
            except Exception as _jinja_err:
                # Log so the user can see WHY the template failed
                logger.warning(
                    f"Jinja2 template rendering failed: {_jinja_err!r}. "
                    f"Falling back to generic 'Role: Content' format."
                )
                pass  # fall through

        # Generic fallback 
        # Used only when no tokenizer and no template are available. Tool schemas have no native rendering here, so inject them as a text block into the system message.
        # May produce incorrect results for models with specific chat formats.
        if tools:
            msgs = _inject_tools_into_system(msgs, tools)
        return self._generic(msgs)

    # Helpers 

    def _render_jinja(
        self,
        messages: List[Dict],
        template: str,
        add_generation_prompt: bool,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        try:
            from jinja2 import Environment, BaseLoader
        except ImportError:
            raise ImportError(
                "jinja2 is required for Jinja2 template rendering.\n"
                "Install it: pip install jinja2\n"
                "Or install transformers which includes it: pip install transformers"
            )

        def raise_exception(message: str):
            """Registered so HuggingFace templates that call raise_exception() work."""
            raise ValueError(message)

        def strftime_now(fmt: str) -> str:
            """Registered so HuggingFace templates that call strftime_now() work."""
            return datetime.datetime.now().strftime(fmt)

        env = Environment(loader=BaseLoader())
        # Register globals that HuggingFace chat templates expect
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"]    = strftime_now

        tmpl = env.from_string(template)
        return tmpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _generic(self, messages: List[Dict]) -> str:
        """Simple Role: Content fallback."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg.get("content") or ""
            if msg["role"] == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                parts.append(f"Tool Result (id={tool_call_id}): {content}")
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                tc_str = json.dumps(msg["tool_calls"])
                parts.append(f"{role}: <tool_call>{tc_str}</tool_call>")
            else:
                parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # Genie incremental helpers 

    def get_genie_delta(
        self,
        new_user_message: str,
        prev_assistant_response: str,
        tokenizer: Optional[Any] = None,
        template: Optional[str] = None,
        is_first_turn: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build the prompt DELTA for Genie multi-turn KV-cache inference.

        Genie's GenieDialog maintains KV cache internally, so we only pass
        the NEW content each turn (not the full history).

          First turn:  system + user message + open assistant turn
          Later turns: close previous assistant turn + user message + open assistant turn

        The template is sourced from tokenizer_config.json (auto-detected by
        genie_plugin.py) — no hardcoding. Works for any model automatically.

        Args:
            new_user_message:        The new user input.
            prev_assistant_response: The assistant's previous response (empty on first turn).
            tokenizer:               Optional HF tokenizer (takes priority).
            template:                Jinja2 template string (from tokenizer_config.json).
            is_first_turn:           True if this is the first turn.
            system_prompt:           System prompt (used on first turn only).
            tools:                   Optional tool schemas (forwarded to format()).

        Returns:
            Prompt delta string to pass to GenieDialog.query().
        """
        sys_prompt = system_prompt or self.DEFAULT_SYSTEM

        if is_first_turn:
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": new_user_message},
            ]
            return self.format(
                msgs,
                tokenizer=tokenizer,
                template=template,
                default_system=sys_prompt,
                add_generation_prompt=True,
                tools=tools,
            )


        prev_msgs = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": ""},          # placeholder
            {"role": "assistant", "content": prev_assistant_response},
        ]
        next_msgs = prev_msgs + [{"role": "user", "content": new_user_message}]

        prev_prompt = self.format(
            prev_msgs,
            tokenizer=tokenizer,
            template=template,
            default_system=sys_prompt,
            add_generation_prompt=False,
        )
        next_prompt = self.format(
            next_msgs,
            tokenizer=tokenizer,
            template=template,
            default_system=sys_prompt,
            add_generation_prompt=True,
            tools=tools,
        )

        # The delta is everything after the previous prompt
        if next_prompt.startswith(prev_prompt):
            return next_prompt[len(prev_prompt):]

        # Fallback: return the full next prompt
        return next_prompt

def _inject_tools_into_system(messages: List[Dict], tools: List[Dict]) -> List[Dict]:
    """
    Inject tool schemas into the system message as a JSON block.

    Used by the Jinja2 and generic-fallback paths when the HF tokenizer is not available (which natively supports tools= in apply_chat_template).
    The format mirrors what most tool-calling fine-tuned models expect:

        Available tools:
        ```json
        [{"type": "function", "function": {...}}, ...]
        ```
    """
    tools_block = (
        "\n\nAvailable tools:\n```json\n"
        + json.dumps(tools, indent=2)
        + "\n```"
    )
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            msgs[i] = dict(m, content=(m.get("content") or "") + tools_block)
            return msgs
    # No system message found — prepend one
    msgs.insert(0, {"role": "system", "content": tools_block.lstrip()})
    return msgs