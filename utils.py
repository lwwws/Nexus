"""
Nexus Utilities Module

This module handles the data ingestion and normalization layer of Nexus.
Its primary responsibility is converting raw, unstructured text exports (specifically from WhatsApp)
into structured Pandas DataFrames that the application can process.

Key Features:
1. Format Auto-Detection: Uses heuristic regex matching to identify which timestamp format
   a specific export is using (iOS vs Android, 12hr vs 24hr, etc.).
2. Multi-line Handling: Reconstructs messages that span multiple lines in the text file.
3. System Message Parsing: Distinguishes between human messages and system notifications
   (e.g., "Messages are end-to-end encrypted").
"""

import re
import datetime
from typing import Dict

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def infer_datetime_format(sample_text):
    """
    Auto-detects the WhatsApp date format from a sample text.
    """
    # Regex definitions
    patterns = {
        # Ex: [13:55, 1/15/2026] - (Brackets, Time first)
        'brackets': r'\[\d{1,2}:\d{2},\s\d{1,2}/\d{1,2}/\d{2,4}\]\s',

        # Ex: 2026/01/14, 21:00:35 - (Year-First, 24hr with seconds)
        'iso_sec': r'(?<!\d)\d{4}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}:\d{2}\s-\s',

        # Ex: 24/01/2020, 8:25 pm - (Day/Month/Year, 12hr)
        '12hr': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',

        # Ex: 26/01/2020, 4:19:30 pm - (Day/Month/Year, 12hr with seconds)
        '12hr_sec': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[APap][mM]\s-\s',

        # Ex: 25/12/2025, 23:58 - (Day/Month/Year, 24hr)
        '24hr': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',

        # Ex: 25/12/2025, 23:58:00 - (Day/Month/Year, 24hr with seconds)
        '24hr_sec': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s-\s'
    }

    # Count matches for each pattern
    matches = {}
    for key, regex in patterns.items():
        matches[key] = len(re.findall(regex, sample_text))

    # Get the key with the most matches (if any)
    best_match = max(matches, key=matches.get)

    if matches[best_match] == 0:
        raise ValueError("Could not detect a valid WhatsApp date format in the file.")

    return best_match

def raw2df(file, key='auto'):
    """
    Converts raw .txt file into a Data Frame

    Args:
        file: Path to the WhatsApp chat export file
        key: Format key - 'auto' (default), '12hr', '24hr', or 'custom'

    Returns:
        pd.DataFrame: DataFrame with columns: date_time, user, message
    """

    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content:
        raise ValueError("File is empty.")

    # Auto-detect format if key is 'auto'
    if key == 'auto':
        key = infer_datetime_format(content[:5000])
        logger.info(f"Auto-detected format: {key}")

    split_formats = {
        'brackets': r'\[\d{1,2}:\d{2},\s\d{1,2}/\d{1,2}/\d{2,4}\]\s',
        'iso_sec': r'(?<!\d)\d{4}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}:\d{2}\s-\s',
        '12hr': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '12hr_sec': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[APap][mM]\s-\s',
        '24hr': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        '24hr_sec': r'(?<!\d)\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s-\s'
    }

    datetime_formats = {
        'brackets': '[%H:%M, %d/%m/%Y] ',
        'iso_sec': '%Y/%m/%d, %H:%M:%S - ',
        '12hr': '%d/%m/%Y, %I:%M %p - ',
        '12hr_sec': '%d/%m/%Y, %I:%M:%S %p - ',
        '24hr': '%d/%m/%Y, %H:%M - ',
        '24hr_sec': '%d/%m/%Y, %H:%M:%S - '
    }

    # Split and parse
    # Replacing newlines with spaces handles multi-line messages
    raw_string = ' '.join(content.split('\n'))

    # Split by the detected date regex
    user_msg = re.split(split_formats[key], raw_string)[1:]
    date_time = re.findall(split_formats[key], raw_string)

    if len(date_time) != len(user_msg):
        # Fallback if something went wrong, though findall/split usually align
        min_len = min(len(date_time), len(user_msg))
        date_time = date_time[:min_len]
        user_msg = user_msg[:min_len]

    df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})

    # Convert to datetime object
    try:
        df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])
    except ValueError as e:
        print(f"Error parsing datetime: {e}")
        # Fallback: try parsing without strict format (slower but safer)
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

    # Extracting user and message
    usernames = []
    msgs = []

    for i in df['user_msg']:
        # Lazy match for "User Name: Message"
        a = re.split(r'([\w\W]+?):\s', i, maxsplit=1)

        if len(a) > 1:
            # Normal message
            usernames.append(a[1])
            msgs.append(a[2])
        else:
            # System message
            usernames.append("group_notification")
            msgs.append(a[0])

    df['user'] = usernames
    df['message'] = msgs
    df.drop('user_msg', axis=1, inplace=True)

    print(f"Successfully parsed {len(df)} messages.")
    return df


class UserMapper:
    """
    Maps raw user IDs (including 'undefined') to semantic aliases
    like 'Speaker A', 'Speaker B'.
    """

    def __init__(self):
        self.user_map: Dict[str, str] = {}
        self.next_alias_idx = 0

    def get_alias(self, real_user: str) -> str:
        real_user = (real_user or "").strip()

        # Trap "undefined", "null", or empty strings
        if not real_user or real_user.lower() in ["undefined", "null", "none"]:
            return "Unknown Speaker"

        if real_user not in self.user_map:
            # Generate "Speaker A", "Speaker B", etc.
            if self.next_alias_idx < 26:
                suffix = chr(65 + self.next_alias_idx)  # A-Z
            else:
                suffix = str(self.next_alias_idx + 1)  # 27, 28...

            self.user_map[real_user] = f"Speaker {suffix}"
            self.next_alias_idx += 1

        return self.user_map[real_user]