import re
from typing import Tuple

# Common Indian State/UT Codes
KNOWN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# OCR character substitution maps
CHAR_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1', '|': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
    'P': '9'
}

DIGIT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B'
}

class PlateNormalizer:
    """
    Cleans, standardizes, and validates license plate text outputs from OCR models.
    Applies heuristic positional character repair (digits vs letters) and format matching.
    """
    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        if not raw_text:
            return ""
        # Remove whitespace, hyphens, dots, and special characters
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text.strip().upper())
        return cleaned

    @classmethod
    def normalize_plate(cls, raw_text: str, raw_confidence: float) -> Tuple[str, float, bool]:
        """
        Normalizes OCR text output.
        Returns: (normalized_plate_string, adjusted_confidence, is_valid_format)
        """
        text = cls.clean_text(raw_text)
        if len(text) < 4 or len(text) > 12:
            return text, max(0.0, raw_confidence * 0.5), False

        # 1. Attempt standard Indian Plate format rectification:
        # Pattern: [2 Letters: State] [1-2 Digits: RTO] [1-3 Letters: Series] [4 Digits: Number]
        # Length typically 9 or 10 characters (e.g. TN01AB1234 or DL3CAB1234)
        if 8 <= len(text) <= 11:
            corrected = list(text)

            # Ensure first 2 characters are letters
            for i in [0, 1]:
                if corrected[i].isdigit() and corrected[i] in DIGIT_TO_CHAR:
                    corrected[i] = DIGIT_TO_CHAR[corrected[i]]

            # Ensure positions 2 & 3 are digits if length >= 9
            if len(corrected) >= 9:
                if corrected[2].isalpha() and corrected[2] in CHAR_TO_DIGIT:
                    corrected[2] = CHAR_TO_DIGIT[corrected[2]]
                if corrected[3].isalpha() and corrected[3] in CHAR_TO_DIGIT:
                    corrected[3] = CHAR_TO_DIGIT[corrected[3]]

            # Ensure last 4 characters are digits
            for i in range(len(corrected) - 4, len(corrected)):
                if corrected[i].isalpha() and corrected[i] in CHAR_TO_DIGIT:
                    corrected[i] = CHAR_TO_DIGIT[corrected[i]]

            candidate = "".join(corrected)

            # Check standard pattern
            indian_plate_regex = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
            if re.match(indian_plate_regex, candidate):
                state = candidate[:2]
                bonus = 0.15 if state in KNOWN_STATE_CODES else 0.05
                final_conf = min(1.0, raw_confidence + bonus)
                return candidate, round(final_conf, 2), True

        # 2. General alphanumeric plate format (5 to 10 alphanumeric chars)
        general_regex = r'^[A-Z0-9]{5,10}$'
        if re.match(general_regex, text):
            return text, round(raw_confidence, 2), True

        return text, round(raw_confidence * 0.7, 2), False
