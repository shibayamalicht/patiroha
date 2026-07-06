# Copyright 2026 しばやま (shibayamalicht)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text tokenization and normalization for patent documents."""

from patiroha.tokenize.filters import apply_ngram_filters
from patiroha.tokenize.japanese import extract_keywords, tokenize_for_tfidf
from patiroha.tokenize.normalize import normalize_text, strip_html

__all__ = [
    "extract_keywords",
    "tokenize_for_tfidf",
    "normalize_text",
    "strip_html",
    "apply_ngram_filters",
]
