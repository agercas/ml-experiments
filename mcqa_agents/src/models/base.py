from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all models. All models should inherit from this class."""

    @abstractmethod
    def predict(
        self, questions: str | list[str], answers_list: list[str] | list[list[str]], text: str
    ) -> int | list[int]:
        """
        Predict answers for given questions based on the provided text.

        Args:
            questions: Single question string or list of questions
            answers_list: Single list of answer choices or list of answer choice lists
            text: Context text to base predictions on

        Returns:
            Single predicted index or list of predicted indices
        """
        pass
