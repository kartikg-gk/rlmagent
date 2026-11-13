"""Modal picker for switching provider and model."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView

from rlmagent.config.store import PROVIDERS, ProviderInfo, load_config, recent_models


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The result of a completed picker run."""

    provider: str
    label: str
    model: str
    api_key: str | None
    base_url: str | None = None


class _Row(ListItem):
    """A list row carrying an opaque value."""

    def __init__(self, text: str, value: str) -> None:
        super().__init__(Label(text))
        self.value = value


def _explain(error: Exception) -> str:
    """A failed provider check, in words that say what to change."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in (401, 403):
        return f"The provider rejected this key ({status}). Check it and try again."
    if status == 404:
        return "Nothing at that endpoint (404). Check the base URL with Ctrl+E."
    if status:
        return f"The provider answered {status}. Try again, or check the endpoint."
    return f"Could not reach the provider ({type(error).__name__}). Check the endpoint."


class ModelScreen(ModalScreen[ModelChoice | None]):
    """Small centred dialog: pick a provider, a model, and a key if needed."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+f", "discover", "Fetch models"),
        ("ctrl+t", "test", "Test credentials"),
        ("ctrl+e", "endpoint", "Endpoint"),
        ("ctrl+r", "reveal", "Show key"),
        ("ctrl+d", "forget", "Remove recent"),
    ]

    def __init__(self, current_provider: str = "", current_model: str = "") -> None:
        super().__init__()
        self._current_provider = current_provider
        self._fetched: list[str] = []
        self._current_model = current_model
        self._info: ProviderInfo | None = None
        self._model: str = ""
        self._input_mode = "key"
        self._endpoint = None

    # -- composition --------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Label("Select provider", id="model-title")
            yield ListView(id="model-list")
            yield Input(placeholder="Type to filter models", id="model-filter")
            yield Input(placeholder="API key", password=True, id="model-key")
            yield Label("", id="model-hint")

    def on_mount(self) -> None:
        self.query_one("#model-key", Input).display = False
        self.query_one("#model-filter", Input).display = False
        self._show_providers()

    # -- steps --------------------------------------------------------------

    def _show_providers(self) -> None:
        """Step 1 — list the supported providers."""
        self.query_one("#model-title", Label).update("Select provider")
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        for recent in recent_models():
            listing.append(_Row(f"↺ {recent['model']} · {recent['provider']}",
                                f"recent:{recent['provider']}|{recent['model']}"))
        for info in PROVIDERS:
            marker = "●" if info.key == self._current_provider else " "
            listing.append(_Row(f"{marker} {info.label}", info.key))
        listing.index = 0
        listing.focus()

    def _show_models(self, info: ProviderInfo) -> None:
        """Step 2 — list that provider's models."""
        self._info = info
        self.query_one("#model-title", Label).update(f"{info.label} — select model")
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        from rlmagent.model.discovery import price_label

        for name in info.models:
            marker = "●" if name == self._current_model else " "
            price = price_label(info.key, name)
            listing.append(_Row(f"{marker} {name}" + (f"   {price}" if price else ""), name))
        listing.append(_Row("Enter a model identifier…", "__custom__"))
        listing.index = 0
        listing.focus()
        self.query_one("#model-hint", Label).update(
            "Ctrl+F fetch models · Ctrl+T test credentials · Ctrl+E endpoint"
        )

    def _input(self, mode, title, placeholder, password=False):
        self._input_mode = mode
        self.query_one("#model-list", ListView).display = False
        field = self.query_one("#model-key", Input)
        field.password, field.placeholder, field.value = password, placeholder, ""
        field.display = True
        self.query_one("#model-title", Label).update(title)
        field.focus()

    async def _remote(self, show_models):
        if self._info is None:
            return
        from rlmagent.model.discovery import discover_models, check_provider

        hint = self.query_one("#model-hint", Label)
        hint.update("Checking provider…")
        try:
            if not show_models:
                hint.update(await check_provider(self._info.key, base_url=self._endpoint))
                return
            models = await discover_models(self._info.key, base_url=self._endpoint)
        except Exception as error:
            hint.update(_explain(error))
            return
        hint.update(f"Model endpoint accepted credentials; {len(models)} models returned.")
        if show_models:
            self._fetched = models
            self._list_fetched("")
            box = self.query_one("#model-filter", Input)
            box.value, box.display = "", True
            box.focus()

    def _list_fetched(self, text: str) -> None:
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        from rlmagent.model.discovery import price_label

        rows = [model for model in self._fetched if text.lower() in model.lower()]
        provider = self._info.key if self._info is not None else ""
        for model in rows:
            price = price_label(provider, model)
            listing.append(_Row(f"{model}   {price}" if price else model, model))
        listing.append(_Row("Enter a model identifier…", "__custom__"))
        listing.index = rows.index(self._current_model) if self._current_model in rows else 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-filter":
            event.stop()
            self._list_fetched(event.value)

    def action_forget(self):
        from rlmagent.config.store import forget_model

        row = self.query_one("#model-list", ListView).highlighted_child
        value = getattr(row, "value", "")
        if self._info is None and value.startswith("recent:"):
            forget_model(*value.removeprefix("recent:").split("|", 1))
            self._show_providers()

    def action_reveal(self):
        field = self.query_one("#model-key", Input)
        field.password = not field.password

    def action_discover(self):
        self.run_worker(self._remote(True), exclusive=True, group="provider-check")

    def action_test(self):
        self.run_worker(self._remote(False), exclusive=True, group="provider-check")

    def action_endpoint(self):
        if self._info is not None and self._info.key not in ("vertex", "codex", "copilot", "claude-subscription", "xai-subscription"):
            self._input("endpoint", "Compatible API base URL", "https://host/v1")

    def _show_key_prompt(self) -> None:
        """Step 3 — ask for an API key, only when none is stored."""
        assert self._info is not None
        self._input_mode = "key"
        self.query_one("#model-key", Input).password = True
        self.query_one("#model-list", ListView).display = False
        field = self.query_one("#model-key", Input)
        field.display = True
        field.focus()
        self.query_one("#model-title", Label).update(f"{self._info.label} — API key")
        self.query_one("#model-hint", Label).update("Enter to save · Esc to cancel")

    def _stored_key(self, info: ProviderInfo) -> str | None:
        """Key already saved in config or the environment, if any."""
        import os

        return os.environ.get(info.key_env) or load_config().api_keys.get(info.key)

    # -- events -------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        row = event.item
        if not isinstance(row, _Row):
            return
        if row.value.startswith("recent:"):
            provider, _, model = row.value.removeprefix("recent:").partition("|")
            self._info = next((p for p in PROVIDERS if p.key == provider), None)
            if self._info is None:
                self._show_providers()
                return
            self._on_model_chosen(model)
        elif self._info is None:
            self._on_provider_chosen(row.value)
        else:
            self._on_model_chosen(row.value)

    def _on_provider_chosen(self, key: str) -> None:
        info = next((p for p in PROVIDERS if p.key == key), None)
        if info is None:
            return
        self._show_models(info)

    def _on_model_chosen(self, model: str) -> None:
        assert self._info is not None
        if model == "__custom__":
            self._input("model", "Model identifier", "provider/model")
            return
        self._model = model
        if not self._info.needs_key or self._stored_key(self._info):
            self._finish(None)
            return
        self._show_key_prompt()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if event.input.id == "model-filter":
            listing = self.query_one("#model-list", ListView)
            event.input.display = False
            listing.focus()
            if listing.highlighted_child is not None:
                self._on_model_chosen(listing.highlighted_child.value)
            return
        key = event.value.strip()
        if self._input_mode in ("model", "endpoint"):
            if not key:
                return
            if self._input_mode == "endpoint":
                from urllib.parse import urlsplit

                parsed = urlsplit(key)
                if (
                    parsed.scheme not in ("http", "https")
                    or not parsed.netloc
                    or parsed.username
                    or parsed.password
                ):
                    self.query_one("#model-hint", Label).update(
                        "Enter an HTTP(S) base URL without credentials."
                    )
                    return
                self._endpoint = key.rstrip("/")
            self.query_one("#model-key", Input).display = False
            self.query_one("#model-list", ListView).display = True
            self._show_models(self._info)
            if self._input_mode == "model":
                self._on_model_chosen(key)
            return
        if not key:
            self.query_one("#model-hint", Label).update("API key must not be empty.")
            return
        self._finish(key)

    def _finish(self, api_key: str | None) -> None:
        assert self._info is not None
        self.dismiss(
            ModelChoice(
                provider=self._info.key,
                label=self._info.label,
                model=self._model,
                api_key=api_key,
                base_url=self._endpoint,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ModelChoice", "ModelScreen"]
