from enum import Enum


class ProjectDataSetTags(Enum):
    TRACES = 'traces'
    PLAIN_TEXT = 'metadata/plain_text'
    CIPHER_TEXT = 'metadata/cipher_text'
    KEY = 'metadata/key'
