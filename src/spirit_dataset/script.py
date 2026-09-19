from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from spirit_dataset.text_cleaning import is_quoted_utterance, unquote_utterance

SPEECH_UI_TYPES: frozenset[str] = frozenset({"normal", "typing"})
SITUATION_UI_TYPES: frozenset[str] = frozenset({"narration", "place"})
PRESENT_UI_TYPE = "present"
CHOICE_UI_TYPE = "choice"
PLAYER_UI_TYPE = "player"
EARLIER_EXCHANGE_LIMIT = 3

type ExchangeHistory = tuple[tuple[tuple[str, ...], str], ...]
CONVERTED_UI_TYPES: frozenset[str] = frozenset(
    {*SPEECH_UI_TYPES, *SITUATION_UI_TYPES, PRESENT_UI_TYPE, CHOICE_UI_TYPE, PLAYER_UI_TYPE}
)


class ChoiceLayout(StrEnum):
    EACH_ROW_IS_OPTION = "each_row_is_option"
    GROUPED_BY_CHOICE_GROUP = "grouped_by_choice_group"


@dataclass(frozen=True)
class ScriptLine:
    key: int
    talk_index: int
    ui_type: str
    choice_group: int
    speaker_no: int
    speaker_name: str | None
    text: str


@dataclass(frozen=True)
class ScriptExchange:
    user_items: tuple[str, ...]
    previous_spirit_text: str | None
    spirit_lines: tuple[ScriptLine, ...]
    context_keys: tuple[int, ...]
    previous_user_items: tuple[str, ...] = ()
    earlier_exchanges: ExchangeHistory = ()

    @property
    def spirit_text(self) -> str:
        return " ".join(line.text for line in self.spirit_lines)

    @property
    def keys(self) -> tuple[int, ...]:
        return (*self.context_keys, *(line.key for line in self.spirit_lines))


@dataclass
class PendingVariant:
    items: list[str]
    keys: list[int]


@dataclass(frozen=True)
class ScriptParseResult:
    exchanges: list[ScriptExchange]
    unanswered: list[PendingVariant]
    unconverted: list[ScriptLine]


@dataclass(frozen=True)
class ScriptState:
    items: tuple[str, ...] = ()
    keys: tuple[int, ...] = ()
    previous: str | None = None
    buffer: tuple[ScriptLine, ...] = ()
    previous_items: tuple[str, ...] = ()
    earlier_exchanges: ExchangeHistory = ()


def unique_states(states: Iterable[ScriptState]) -> list[ScriptState]:
    unique: dict[ScriptState, ScriptState] = {}
    for state in states:
        unique.setdefault(replace(state, earlier_exchanges=()), state)
    return list(unique.values())


def choice_item(text: str) -> str:
    return unquote_utterance(text) if is_quoted_utterance(text) else f"({text})"


def context_item(line: ScriptLine) -> str | None:
    if line.ui_type in SITUATION_UI_TYPES:
        return f"({line.text})"
    if line.ui_type == PRESENT_UI_TYPE:
        return f"(선물: {line.text})"
    if line.ui_type == PLAYER_UI_TYPE:
        return line.text
    if line.ui_type in SPEECH_UI_TYPES:
        return f"{line.speaker_name}: {line.text}" if line.speaker_name else f"({line.text})"
    return None


def parse_choice_options(
    block: Sequence[ScriptLine], layout: ChoiceLayout
) -> list[tuple[int, str, tuple[int, ...]]]:
    choices = [line for line in block if line.ui_type == CHOICE_UI_TYPE]
    if layout is ChoiceLayout.EACH_ROW_IS_OPTION:
        return [
            (index, unquote_utterance(line.text), (line.key,)) for index, line in enumerate(choices)
        ]
    grouped: dict[int, list[ScriptLine]] = {}
    for line in choices:
        grouped.setdefault(line.choice_group, []).append(line)
    return [
        (
            choice_group,
            " ".join(choice_item(line.text) for line in lines),
            tuple(line.key for line in lines),
        )
        for choice_group, lines in grouped.items()
    ]


class ScriptWalker:
    def __init__(self, is_spirit_line: Callable[[ScriptLine], bool], layout: ChoiceLayout) -> None:
        self._is_spirit_line = is_spirit_line
        self._layout = layout

    def exchanges(self, lines: Sequence[ScriptLine]) -> list[ScriptExchange]:
        return self.parse(lines).exchanges

    def parse(self, lines: Sequence[ScriptLine]) -> ScriptParseResult:
        converted = [line for line in lines if line.ui_type in CONVERTED_UI_TYPES and line.text]
        unconverted = [
            line for line in lines if line.ui_type not in CONVERTED_UI_TYPES or not line.text
        ]
        exchanges: list[ScriptExchange] = []
        unanswered: list[PendingVariant] = []
        states = self._walk(converted, [ScriptState()], exchanges)
        for state in states:
            if not state.buffer and state.items:
                unanswered.append(PendingVariant(list(state.items), list(state.keys)))
            self._flush(state, exchanges)
        unique: dict[
            tuple[tuple[str, ...], tuple[str, ...], str | None, tuple[int, ...]], ScriptExchange
        ] = {}
        for exchange in exchanges:
            unique.setdefault(
                (
                    exchange.user_items,
                    exchange.previous_user_items,
                    exchange.previous_spirit_text,
                    tuple(line.key for line in exchange.spirit_lines),
                ),
                exchange,
            )
        return ScriptParseResult(list(unique.values()), unanswered, unconverted)

    def _flush(
        self,
        state: ScriptState,
        exchanges: list[ScriptExchange],
    ) -> ScriptState:
        if not state.buffer:
            return state
        exchanges.append(
            ScriptExchange(
                user_items=state.items,
                previous_spirit_text=state.previous,
                spirit_lines=state.buffer,
                context_keys=state.keys,
                previous_user_items=state.previous_items,
                earlier_exchanges=state.earlier_exchanges,
            )
        )
        earlier = (
            state.earlier_exchanges
            if state.previous is None
            else (*state.earlier_exchanges, (state.previous_items, state.previous))
        )
        return ScriptState(
            previous=" ".join(line.text for line in state.buffer),
            previous_items=state.items,
            earlier_exchanges=earlier[-EARLIER_EXCHANGE_LIMIT:],
        )

    def _walk(
        self,
        lines: Sequence[ScriptLine],
        states: list[ScriptState],
        exchanges: list[ScriptExchange],
    ) -> list[ScriptState]:
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.ui_type == CHOICE_UI_TYPE:
                states = unique_states(self._flush(state, exchanges) for state in states)
                block_end = index
                while block_end < len(lines) and lines[block_end].ui_type in (
                    CHOICE_UI_TYPE,
                    PLAYER_UI_TYPE,
                ):
                    block_end += 1
                options = parse_choice_options(lines[index:block_end], self._layout)
                branch_end = block_end
                if self._layout is ChoiceLayout.GROUPED_BY_CHOICE_GROUP:
                    while (
                        branch_end < len(lines)
                        and lines[branch_end].choice_group > 0
                        and lines[branch_end].ui_type != CHOICE_UI_TYPE
                    ):
                        branch_end += 1
                branch_lines = lines[block_end:branch_end]
                joined: list[ScriptState] = []
                for option_group, option_text, option_keys in options:
                    option_states = [
                        ScriptState(
                            items=(*state.items, option_text),
                            keys=(*state.keys, *option_keys),
                            previous=state.previous,
                            previous_items=state.previous_items,
                            earlier_exchanges=state.earlier_exchanges,
                        )
                        for state in states
                    ]
                    if branch_lines:
                        option_states = self._walk(
                            [
                                branch_line
                                for branch_line in branch_lines
                                if branch_line.choice_group == option_group
                            ],
                            option_states,
                            exchanges,
                        )
                    joined.extend(option_states)
                states = unique_states(joined)
                index = branch_end
                continue
            if line.ui_type in SPEECH_UI_TYPES and self._is_spirit_line(line):
                states = [
                    replace(state, buffer=(*state.buffer, line))
                    for state in states
                ]
            else:
                item = context_item(line)
                if item is not None:
                    states = unique_states(
                        replace(
                            flushed,
                            items=(*flushed.items, item),
                            keys=(*flushed.keys, line.key),
                        )
                        for state in states
                        for flushed in (self._flush(state, exchanges),)
                    )
            index += 1
        return states
