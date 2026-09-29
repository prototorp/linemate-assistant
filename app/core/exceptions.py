"""Small exception hierarchy so loaders can report which row/file is bad and carry on"""


class DocumentLoadError(Exception):
    """A file could not be turned into a Document."""


class UnsupportedFileTypeError(DocumentLoadError):
    """A file in the docs folder is not a supported type (only .md)."""


class TicketLoadError(Exception):
    """A tickets.csv row could not be turned into a Ticket."""


class CrewLoadError(Exception):
    """A crew.csv row could not be turned into a CrewMember."""


class InvalidReferenceError(Exception):
    """A request pointed at a crew member / document / ticket that does not exist."""


class AnswerBackendError(Exception):
    """The local LLM / embedding backend (Ollama) could not be reached or is missing a model."""