import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_embedding_coverage.py")
DEDICATED = [
    "ck-embed/src/lib.rs",
    "ck-embed/src/mixedbread.rs",
    "ck-embed/src/reranker.rs",
    "ck-embed/src/tokenizer.rs",
    "ck-models/src/lib.rs",
    "ck-engine/src/semantic_v3.rs",
]


def report_for(percent=80.0):
    files = []
    for path in DEDICATED + ["ck-index/src/lib.rs"]:
        files.append(
            {
                "filename": f"/repo/{path}",
                "summary": {
                    "lines": {"covered": percent, "count": 100, "percent": percent}
                },
            }
        )
    return {"data": [{"files": files}]}


class CoverageGateTests(unittest.TestCase):
    def run_report(self, report, *args):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(report, file)
            file.flush()
            return subprocess.run(
                [sys.executable, str(SCRIPT), *args, file.name],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_report_mode_prints_dedicated_and_shared_results(self):
        result = self.run_report(report_for(), "--report")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ck-embed/src/lib.rs", result.stdout)
        self.assertIn("ck-index/src/lib.rs", result.stdout)

    def test_check_mode_rejects_a_dedicated_file_below_eighty_percent(self):
        report = report_for()
        report["data"][0]["files"][0]["summary"]["lines"]["percent"] = 79.9

        result = self.run_report(report, "--check")

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("ck-embed/src/lib.rs", result.stderr)

    def test_check_mode_rejects_missing_dedicated_file(self):
        report = report_for()
        report["data"][0]["files"] = report["data"][0]["files"][1:]

        result = self.run_report(report, "--check")

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("ck-embed/src/lib.rs", result.stderr)


if __name__ == "__main__":
    unittest.main()
