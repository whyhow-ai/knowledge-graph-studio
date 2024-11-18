"""Document processors."""

import json
import logging
import os
import re
import string
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from whyhow_api.models.common import DatasetModel, PDFProcessorConfig

logger = logging.getLogger(__name__)


class GeneralProcessor:
    """General processor class."""

    def __init__(self, config_model: BaseSettings, **kwargs: Any) -> None:
        try:
            self.config = config_model(**kwargs)  # type: ignore
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    def process(self) -> None:
        """Process the data."""
        raise NotImplementedError("Subclasses should implement this method.")


class Dataset:
    """Dataset class."""

    def __init__(self, dataset: Union[Dict[str, List[str]], List[str]]):
        self.model = DatasetModel(dataset=dataset)

    def is_list(self) -> bool:
        """Check if the dataset is a list."""
        return isinstance(self.model.dataset, List)

    def is_dict(self) -> bool:
        """Check if the dataset is a dictionary."""
        return isinstance(self.model.dataset, Dict)

    def save(
        self, directory: str | None = None, filename: str | None = None
    ) -> None:
        """
        Save.

        Parameters
        ----------
        directory : str, optional
            Directory, by default None

        filename : str, optional
            Filename, by default None
        """
        if directory is None:
            directory = os.getcwd()
        else:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

        if filename is None:
            filename = "dataset.json"

        if filename.endswith(".json") is None:
            raise RuntimeError('Filename must end with ".json"')

        with open(
            os.path.join(directory, filename), "w", encoding="utf-8"
        ) as f:
            json.dump(self.model.dataset, f, ensure_ascii=False, indent=2)

    def __getitem__(self, key: Any) -> Any:
        """Get item."""
        return self.model.dataset[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item."""
        self.model.dataset[key] = value

    def __delitem__(self, key: Any) -> None:
        """Delete item."""
        del self.model.dataset[key]

    def __iter__(self) -> Iterable[Any]:
        """Iterate."""
        return iter(self.model.dataset)

    def __len__(self) -> int:
        """Get length."""
        return len(self.model.dataset)

    def __repr__(self) -> str:
        """Get representation."""
        return repr(self.model.dataset)


class PDFProcessor(GeneralProcessor):
    """PDF processor class."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PDFProcessorConfig, **kwargs)  # type: ignore[arg-type]

    def extract_chunks(self, limit: int | None = None) -> List[str]:
        """
        Extract chunks.

        Process a PDF file and split its content into chunks.

        Parameters
        ----------
        limit : int, optional
            The maximum number of chunks to return, by default None

        Returns
        -------
        List[str]
            A list of strings, each representing a chunk of the PDF content.
        """
        try:
            # create a loader
            loader = PyPDFLoader(self.config.file_path)
            # load your data
            data = loader.load()

            if data is None:
                # TODO: Handle more elegantly.
                raise RuntimeError(
                    "No data laoded from the file, or the file is empty."
                )

            # Split your data up into smaller documents with Chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            # Convert Document objects into strings and return the list
            chunks = [
                str(doc.page_content)
                for doc in text_splitter.split_documents(data)
            ]
            # If limit is specified and less than the length of chunks,
            # return a slice up to limit
            if limit is not None and limit < len(chunks):
                return chunks[:limit]
            # Otherwise, return all chunks
            return chunks

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return []

    def process(  # type: ignore[override]
        self,
        limit: int | None = None,
        cleaning_function: Callable[[str], str] | None = None,
        aggregation_function: Callable[[List[str]], Any] | None = None,
    ) -> Dataset:
        """
        Process.

        Processes a PDF document by initially chunking its content, optionally
        applying a cleaning function to each chunk, and finally optionally
        aggregating the processed chunks.

        The processing workflow is as follows:
        1. The document is split into chunks.
        2. If provided, each chunk is cleaned using the `cleaning_function`.
        3. If an `aggregation_function` is provided, it is applied to the
        (optionally cleaned) list of chunks to produce a structured output.
        Otherwise, the processed list of chunks is returned directly.

        Parameters
        ----------
        limit : int, optional
            The maximum number of chunks to process, by default None

        cleaning_function : Callable[[str], str] | None, optional
            A callable that takes a single string argument (a chunk of text)
            and returns a processed string. This function is applied to each
            chunk of the PDF content. If `None`, no cleaning is performed,
            by default None

        aggregation_function : Callable[[List[str]], Any] | None, optional
            A callable that takes a list of strings (the chunks, after
            cleaning) and returns a structured aggregation of these chunks.
            The structure of the aggregation depends on the implementation
            of this function.
            If `None`, the chunks are returned as a list without further
            processing, by default None

        Returns
        -------
        Dataset
            If `aggregation_function` is provided, returns a Dataset instance
            containing the result of the aggregation function. Otherwise,
            returns a Dataset instance containing the processed list of chunks.

        Raises
        ------
        Exception
            Any exceptions raised by `extract_chunks`, `cleaning_function`, or
            `aggregation_function` will propagate to the caller.

        Notes
        -----
        TODO: Make the `_functions` have an ABC interface to adhere to.
        """
        chunks = self.extract_chunks(limit=limit)

        # Sanitize chunks
        chunks = [str(chunk) for chunk in chunks]

        logger.info(f"Extracted: {len(chunks)} chunks")

        if cleaning_function:
            chunks = [cleaning_function(c) for c in chunks]

        if aggregation_function:
            return Dataset(aggregation_function(chunks))
        return Dataset(chunks)


class HarryPotterProcessingUtils:
    """Notes when used in conjunction with Processor on HP book.

    - :warning: This code treats the extracted sentences verbatim.
        No combination and segmentation is currently performed.
    - No validation of sentence quality is performed.
    - Currently does not extract all chapters - probably due to low quality
        input PDF.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a given text string.

        By removing non-printable characters and reducing
        all sequences of whitespace characters to a single space.

        This function ensures that the text is suitable for
        further processing, particularly useful when preparing
        text for operations that require standardized whitespace
        handling, such as text parsing.

        Parameters
        ----------
        text : str
            The text to clean. This text may contain non-printable characters
            and sequences of whitespace.

        Returns
        -------
        str
            The cleaned text with only printable characters and single spaces
            between words.
        """
        try:
            # Filter to keep only printable characters
            printable = set(string.printable)
            cleaned_text = "".join(filter(lambda x: x in printable, text))

            # Replace new line characters with whitespace
            cleaned_text = cleaned_text.replace("\n", " ")

            # Replace multiple whitespaces with a single space and trim
            # leading/trailing spaces
            return re.sub(r"\s+", " ", cleaned_text).strip()
        except Exception as e:
            raise ValueError(f"Error cleaning text: {e}")

    @staticmethod
    def aggregate_chapters(
        book_lines: List[str], return_chapter_count: bool = False
    ) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], int]]:
        """
        Aggregate lines from a book into chapters.

        Based on "CHAPTER" headers and counts the number of chapters.
        Lines before the first chapter are aggregated under a special
        key 'PROLOGUE'.

        Parameters
        ----------
        book_lines : List[str]
            The lines of the book as a list of strings.

        return_chapter_count : bool, optional
            Whether to return the chapter count, by default False

        Returns
        -------
        Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], int]]
            Either a a tuple containing a dictionary with chapter names as
            keys and a list of lines as values, and an integer representing
            the number of chapters extracted, or just the chapter names in a
            dictionary with their respective chunks as a list of strings.
        """
        try:
            # Regular expression to match "CHAPTER" followed by the
            # chapter number in words
            chapter_pattern = re.compile(r"CHAPTER\s+(\w+)\s*", re.IGNORECASE)
            chapters: dict[str, list[Any]] = {"PROLOGUE": []}
            current_chapter = "PROLOGUE"

            for line in book_lines:
                chapter_match = chapter_pattern.match(line)
                if chapter_match:
                    # Extract the chapter number (name) from the match
                    chapter_name = chapter_match.group(1).upper()
                    # Correctly format the chapter name
                    current_chapter = f"CHAPTER {chapter_name}"
                    # Initialize a new list in the dictionary for this chapter
                    # if it doesn't already exist
                    if current_chapter not in chapters:
                        chapters[current_chapter] = []
                else:
                    # Aggregate lines under the current chapter key
                    chapters[current_chapter].append(line)

            # The number of chapters extracted is the total keys minus one
            # for the 'PROLOGUE'
            chapter_count = (
                len(chapters) - 1 if "PROLOGUE" in chapters else len(chapters)
            )

            if return_chapter_count:
                return chapters, chapter_count
            return chapters
        except Exception as e:
            raise ValueError(f"Error aggregating chapters: {e}")
