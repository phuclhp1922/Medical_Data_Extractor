from dataclasses import dataclass
from pathlib import Path
import os 
import re
import json 

@dataclass(frozen=True) 
class Figure:
    """A data class representing a figure in the corpus.

    Attributes:
        label (str): The extracted label for the figure, if available.
        image_path (Path): The path to the figure's image file.
        caption (str): The caption text for the figure.
        caption_label (str | None): The label extracted from the caption, if available.
    """
    label: str 
    image_path: Path
    caption: str
    caption_label: str | None

    @property
    def disagrees(self) -> bool:
        """Determine if there is a disagreement between the figure's label and its caption label.

        Returns:
            bool: True if both labels are present and they do not match; False otherwise.
        """
        return self.caption_label is not None and self.label != self.caption_label

    @property
    def confirmed(self) -> bool:
        """Determine if the figure's label is confirmed by its caption label.

        Returns:
            bool: True if both labels are present and they match; False otherwise.
        """
        return self.caption_label is not None and self.label == self.caption_label


class CorpusError(Exception):
    """Base class for exceptions in this module."""


def corpus_root() -> Path:
    """Return the root directory of the corpus.

    This function would typically determine the path to the corpus based on configuration
    or environment variables. For the purpose of this example, it returns a placeholder string."""
    return Path(os.environ.get("CURATOR_CORPUS_DIR", "extracted_data"))



def _caption_label(caption: str, chapter: int, image_count: int) -> str | None:
    """Extract the figure label from a caption, if it matches the expected chapter and image count.

    The function looks for a pattern like 'Fig. 24.1' in the caption. It checks that the chapter
    number matches the provided chapter and that the figure number does not exceed the image count.
    If a valid label is found, it is returned; otherwise, None is returned."""

    # Regex to find figure labels like 'Fig. 24.1'
    match = re.search(r"Fig\. (\d+)[.-](\d+)", caption)
    if not match:
        return None

    fig_chapter, fig_number = int(match.group(1)), int(match.group(2))

    # Check if the chapter matches and the figure number is within the image count
    if fig_chapter == chapter and 1 <= fig_number <= image_count:
        return f"Fig. {fig_chapter}.{fig_number}"
    
    return None


def agreement_report(resolved: dict[str, list[Figure]]) -> dict[str, int]:
    """Generate a report on the agreement between different methods of labeling figures.

    This function analyzes the resolved case data and counts the number of cases, figures,
    and how many figures are confirmed by captions. It returns a dictionary with these counts."""
    
    cases = len(resolved)
    figures = sum(len(figures) for figures in resolved.values())
    caption_confirmed = sum(
        1 for figures in resolved.values() for f in figures if f.confirmed
    )
    disagrements = sum(
        1 for figures in resolved.values() for f in figures if f.disagrees
    )
    positional_only = sum(
        1 for figures in resolved.values() for f in figures if f.caption_label is None
    )

    return {
        "cases": cases,
        "figures": figures,
        "caption_confirmed": caption_confirmed,
        "disagreements": disagrements,
        "positional_only": positional_only
    }

def chapter_of(path: Path) -> int:
    """Extract the chapter number from a directory name.

    The chapter number is expected to be the leading integer in the directory name,
    followed by a hyphen. For example, '24---A-14-Year-Old-Boy-from-Tanzania' yields 24."""
    import re

    match = re.match(r"(\d+)-", Path(path).name)
    if not match:
        raise CorpusError(f"Directory name '{path}' does not start with a chapter number.")
    return int(match.group(1)) 


def resolve_case(case_dir: Path) -> list[Figure]:
    """Resolve a case directory into its constituent parts.

    This function would typically read metadata and other relevant information from the case directory.
    For the purpose of this example, it returns a placeholder list of Figure objects."""
    content_dir = Path(f"{case_dir}/auto") 
    chapter = chapter_of(case_dir) 
    files = list(content_dir.glob("*_content_list.json"))

    if not files:
        raise CorpusError(f"No _content_list.json found for {chapter} in {content_dir}")
    
    with open(files[0], 'r', encoding='utf-8') as file:
        data = json.load(file) 

    images = [item for item in data if item.get("type") == "image"] 

    image_count = len(images) 
    figures = []

    for position, item in enumerate(images, start=1):
        caption = " ".join(item.get("image_caption", [])) 
        image_path = content_dir / item.get("img_path", "")
        label = f"Fig. {chapter}.{position}"
        caption_label = _caption_label(caption, chapter, image_count)

        figures.append(Figure(label=label, image_path=image_path, caption=caption, caption_label=caption_label))

    return figures


def resolve_corpus() -> dict[str, list[Figure]]:
    """Resolve the entire corpus into a structured format.

    This function would typically iterate over all case directories in the corpus root,
    resolving each one and aggregating the results. It returns a dictionary."""
    corpus_dir = corpus_root()
    resolved = {}

    case_dirs = [p for p in corpus_dir.iterdir() if p.is_dir()]

    for case_dir in sorted(case_dirs, key=lambda x: chapter_of(x)):   
        figures = resolve_case(case_dir)
        resolved[case_dir.name] = figures

    return resolved