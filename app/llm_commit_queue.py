from typing import Dict, Optional, Tuple


class OrderedCommitQueue:
    def __init__(self):
        self.next_id = 1
        self.pending: Dict[int, Dict[str, str]] = {}

    def add_result(self, sentence_id: int, result: Dict[str, str]):
        self.pending[sentence_id] = result

    def pop_ready(self) -> Optional[Tuple[int, Dict[str, str]]]:
        if self.next_id in self.pending:
            result = self.pending.pop(self.next_id)
            sid = self.next_id
            self.next_id += 1
            return sid, result
        return None
