import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import List

class SemanticSplitter:
    def __init__(self, content: List[str], embedding_model: SentenceTransformer,
                 method: str = "number", partition_count: int = 10, quantile: float = 0.9) -> None:
        """
        Initializes the SemanticSplitter with content and an embedding model.
        
        :param content: A list of text segments (e.g., sentences or paragraphs).
        :param embedding_model: A SentenceTransformer embedding model.
        :param method: Method to determine breakpoints; either 'number' or 'quantiles'.
        :param partition_count: The number of partitions to create (used if method is 'number').
        :param quantile: The quantile threshold for selecting breaks (used if method is 'quantiles').
        """
        self.content: List[str] = content
        self.embedding_model: SentenceTransformer = embedding_model
        self.partition_count: int = partition_count
        self.quantile: float = quantile
        
        # Compute embeddings for the entire content list.
        self.embeddings: List[List[float]] = self.embedding_model.encode(self.content)
        # Compute cosine distances between consecutive embeddings.
        self.distances: List[float] = [
            cosine(self.embeddings[i - 1], self.embeddings[i])
            for i in range(1, len(self.embeddings))
        ]
        self.breaks: List[int] = []
        self.centroids: List[List[float]] = []
        self.load_breaks(method=method)

    def centroid_distance(self, embedding_id: int, centroid_id: int) -> float:
        """
        Computes the cosine distance between a specific embedding and a centroid.
        
        :param embedding_id: Index of the embedding.
        :param centroid_id: Index of the centroid.
        :return: Cosine distance between the embedding and the centroid.
        """
        return cosine(self.embeddings[embedding_id], self.centroids[centroid_id])

    def adjust_neighbors(self) -> None:
        """
        Adjusts breakpoints based on neighbor criteria (if needed).
        
        This method is a placeholder for potential refinement of breakpoints.
        Currently, it resets the breaks.
        """
        self.breaks = []

    def load_breaks(self, method: str = 'number') -> None:
        """
        Loads breakpoints based on the selected method.
        
        :param method: 'number' selects a fixed number of breakpoints,
                       'quantiles' selects breakpoints where distances exceed a threshold.
        """
        if method == 'number':
            # Ensure that partition_count is not greater than the number of distances.
            if self.partition_count > len(self.distances):
                self.partition_count = len(self.distances)
            # Use argpartition to select indices corresponding to the largest distances.
            self.breaks = list(np.sort(np.argpartition(self.distances, self.partition_count - 1)
                                        [:self.partition_count - 1]))
        elif method == 'quantiles':
            threshold = np.quantile(self.distances, self.quantile)
            self.breaks = [i for i, v in enumerate(self.distances) if v >= threshold]
        else:
            self.breaks = []

    def get_centroid(self, beginning: int, end: int) -> List[float]:
        """
        Computes the centroid (embedding) for a chunk of content.
        
        :param beginning: Start index of the chunk.
        :param end: End index of the chunk.
        :return: The centroid embedding for the chunk.
        """
        text_chunk = '\n'.join(self.content[beginning:end])
        return self.embedding_model.encode(text_chunk)

    def load_centroids(self) -> None:
        """
        Computes centroids for each chunk based on the detected breakpoints.
        """
        if len(self.breaks) == 0:
            self.centroids = [self.get_centroid(0, len(self.content))]
        else:
            self.centroids = []
            beginning = 0
            for break_position in self.breaks:
                self.centroids.append(self.get_centroid(beginning, break_position + 1))
                beginning = break_position + 1
            self.centroids.append(self.get_centroid(beginning, len(self.content)))

    def get_chunk(self, beginning: int, end: int) -> str:
        """
        Returns a chunk of content as a single concatenated string.
        
        :param beginning: Start index of the chunk.
        :param end: End index of the chunk.
        :return: The concatenated text chunk.
        """
        return '\n'.join(self.content[beginning:end])

    def get_chunks(self) -> List[str]:
        """
        Splits the content into chunks based on the detected breakpoints.
        
        :return: A list of text chunks.
        """
        if len(self.breaks) == 0:
            return [self.get_chunk(0, len(self.content))]
        else:
            chunks: List[str] = []
            beginning = 0
            for break_position in self.breaks:
                chunks.append(self.get_chunk(beginning, break_position + 1))
                beginning = break_position + 1
            chunks.append(self.get_chunk(beginning, len(self.content)))
            return chunks


