"""Use the existing locked campaign ledger and read only newly appended records."""

from experiments.phase1.budget import Budget as CampaignBudget
import json


class Budget(CampaignBudget):
    def _reload(self):
        if not hasattr(self, "offset"):
            self.offset = 0
            self.pending = {}
            self.settled = {}
            self.lock.timeout = 180
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(0, 2)
            if f.tell() < self.offset:
                raise RuntimeError("Shared ledger was truncated")
            f.seek(self.offset)
            for line in f:
                row = json.loads(line)
                key = row["id"]
                if row["event"] == "reserve":
                    self.pending[key] = row["usd"]
                elif row["event"] == "settle":
                    self.pending.pop(key, None)
                    self.settled[key] = row["usd"]
                else:
                    raise ValueError("Unknown cost event")
            self.offset = f.tell()
