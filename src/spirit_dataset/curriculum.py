from enum import IntEnum

from game_data.story import BOND_STORY_TYPES, MAIN_STORY_TYPE
from spirit_dataset.records import TrainingTask


class SpiritGrade(IntEnum):
    COMMON = 110011
    RARE = 110012
    EPIC = 110014


def learns_narrative(grade: SpiritGrade) -> bool:
    return grade is SpiritGrade.EPIC


def required_tasks(grade: SpiritGrade) -> frozenset[TrainingTask]:
    tasks = {TrainingTask.PERSONA_SPEECH}
    if learns_narrative(grade):
        tasks.add(TrainingTask.SELF_MEMORY)
        tasks.add(TrainingTask.BEHAVIOR_JUDGMENT)
    return frozenset(tasks)


def owns_story(hero_no: int, is_variant: bool, story_type: int, act: int) -> bool:
    if story_type == MAIN_STORY_TYPE:
        return not is_variant
    return story_type in BOND_STORY_TYPES and act == hero_no
