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

"""Statistical analysis utilities for patent data."""

from patiroha.stats.cagr import calculate_cagr
from patiroha.stats.hhi import calculate_diversity, calculate_entropy, calculate_gini, calculate_hhi
from patiroha.stats.representatives import find_representatives, find_representatives_mmr, find_similar

__all__ = [
    "calculate_hhi",
    "calculate_entropy",
    "calculate_gini",
    "calculate_diversity",
    "calculate_cagr",
    "find_representatives",
    "find_representatives_mmr",
    "find_similar",
]
