import os
import pandas as pd
from typing import Any, Dict

class TranscriptExtractor:
    """
    Class to extract transcript data from a file (.vtt or .txt).

    The class automatically detects the file type based on its extension and extracts data accordingly.
    The resulting data is returned as a Pandas DataFrame with the following columns:
      - "id"
      - "speaker"
      - "content"
      - "start"
      - "end"
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes the TranscriptExtractor with the given file path.

        :param file_path: The path to the .vtt or .txt file.
        """
        self.file_path = file_path
        self.file_extension: str = os.path.splitext(file_path)[1].lower()

    def extract(self) -> pd.DataFrame:
        """
        Extracts transcript data from the file based on its extension.

        :return: A Pandas DataFrame containing the extracted data.
        :raises ValueError: If the file extension is not supported.
        """
        if self.file_extension == ".vtt":
            return self._extract_vtt()
        elif self.file_extension == ".txt":
            return self._extract_txt()
        else:
            raise ValueError("Unsupported file extension. Only .vtt and .txt are supported.")

    def _extract_vtt(self) -> pd.DataFrame:
        """
        Extracts data from a .vtt file using the webvtt library.

        :return: A Pandas DataFrame with the extracted transcript data.
        """
        try:
            import webvtt
        except ImportError as e:
            raise ImportError("The 'webvtt' library is required to process .vtt files. Install it via pip.") from e

        data: Dict[str, Any] = {
            "id": [],
            "speaker": [],
            "content": [],
            "start": [],
            "end": []
        }

        for caption in webvtt.read(self.file_path):
            # Split the caption text by colon to separate speaker and content.
            line = caption.text.split(":")
            # Ensure that there are at least two parts.
            while len(line) < 2:
                line = [''] + line
            data["id"].append(caption.identifier)
            data["speaker"].append(line[0].strip())
            data["content"].append(line[1].strip())
            data["start"].append(caption.start)
            data["end"].append(caption.end)

        return pd.DataFrame(data)

    def _extract_txt(self) -> pd.DataFrame:
        """
        Extracts data from a .txt file by reading each non-empty line.
        Each line is treated as content; the other fields are set to empty strings.

        :return: A Pandas DataFrame with the extracted transcript data.
        """
        data: Dict[str, Any] = {
            "id": [],
            "speaker": [],
            "content": [],
            "start": [],
            "end": []
        }

        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()

        for line in lines:
            if line.strip() != "":
                data["id"].append("")
                data["speaker"].append("")
                data["content"].append(line.strip())
                data["start"].append("")
                data["end"].append("")

        return pd.DataFrame(data)


