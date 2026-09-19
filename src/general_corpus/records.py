from collections.abc import Iterator
from typing import Any

from general_corpus.sources import GeneralConversation
from spirit_dataset.judgment import SELF_JUDGMENT_CUE
from spirit_dataset.records import SourceClass, TrainingTask
from spirit_dataset.script import EARLIER_EXCHANGE_LIMIT

HISTORY_TURN_LIMIT = (EARLIER_EXCHANGE_LIMIT + 1) * 2
ROLES = ("user", "assistant")


def _turn_roles(first_role: str, count: int) -> list[str]:
    offset = ROLES.index(first_role)
    return [ROLES[(offset + index) % 2] for index in range(count)]


def _record(
    identifier: str,
    conversation: GeneralConversation,
    task: TrainingTask,
    prompt: list[dict[str, str]],
    completion: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "split": conversation.split,
        "task": task.value,
        "language": conversation.language,
        "source_class": SourceClass.GENERAL_KNOWLEDGE.value,
        "source": {"kind": "general_corpus", "table": conversation.source, "keys": []},
        "event_keys": [f"general:{conversation.conversation_id}"],
        "origin_id": conversation.conversation_id,
        "prompt": [{"role": "system", "content": ""}, *prompt],
        "completion": [{"role": "assistant", "content": completion}],
    }


def training_records(conversation: GeneralConversation) -> Iterator[dict[str, Any]]:
    if conversation.judgment is not None:
        yield _record(
            f"{conversation.conversation_id}:0",
            conversation,
            TrainingTask.GENERAL_JUDGMENT,
            [{"role": "user", "content": f"{SELF_JUDGMENT_CUE}\n{conversation.turns[0]}"}],
            conversation.judgment,
        )
        return
    roles = _turn_roles(conversation.first_role, len(conversation.turns))
    turns = [
        {"role": role, "content": content}
        for role, content in zip(roles, conversation.turns, strict=True)
    ]
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant" or index == 0:
            continue
        history = turns[max(0, index - HISTORY_TURN_LIMIT) : index]
        if history[0]["role"] != "user":
            history = history[1:]
        if not history or history[-1]["role"] != "user":
            continue
        yield _record(
            f"{conversation.conversation_id}:{index}",
            conversation,
            TrainingTask.GENERAL_DIALOGUE,
            history,
            turn["content"],
        )
