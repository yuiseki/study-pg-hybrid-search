import os
import subprocess
import time
import unittest
from pathlib import Path

import psycopg

from tests.e2e.mock_ollama import MODEL_DIMS, MockOllamaServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DSN = "postgresql://postgres:postgres@localhost:5434/sandbox"


def run(cmd, env=None, **kwargs):
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        **kwargs,
    )


def wait_for_db(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with psycopg.connect(DB_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("database did not become ready in time")


class TestEndToEndWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            run(["make", "stop"])
        except subprocess.CalledProcessError:
            pass

        run(["make", "clean"])
        run(["make", "start"])
        wait_for_db()

        cls.mock_ollama = MockOllamaServer()
        cls.mock_ollama.start()

        cls.base_env = os.environ.copy()
        cls.base_env.update(
            {
                "DATABASE_URL": DB_DSN,
                "OLLAMA_URL": cls.mock_ollama.url,
            }
        )

    @classmethod
    def tearDownClass(cls):
        cls.mock_ollama.stop()
        run(["make", "stop"])

    def test_full_flow(self):
        title = "E2E猫"
        body = "テスト用の本文です"

        run(
            [
                "python",
                "scripts/add_document.py",
                "--title",
                title,
                "--body",
                body,
                "--source",
                "e2e",
                "--model",
                "nomic-embed-text:v1.5",
            ],
            env=self.base_env,
        )

        run(
            [
                "python",
                "scripts/embed_documents.py",
                "--model",
                "qwen3-embedding:4b",
                "--batch-size",
                "2",
            ],
            env=self.base_env,
        )

        vec_result = run(
            [
                "python",
                "scripts/search_vector.py",
                "猫",
                "--model",
                "nomic-embed-text:v1.5",
                "--limit",
                "3",
            ],
            env=self.base_env,
        )
        self.assertIn("cos_sim", vec_result.stdout)

        hybrid_result = run(
            [
                "python",
                "scripts/search_hybrid.py",
                "猫",
                "--model",
                "nomic-embed-text:v1.5",
                "--limit",
                "5",
            ],
            env=self.base_env,
        )
        self.assertIn("rrf", hybrid_result.stdout)

        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.title, e.model, e.dims
                    FROM documents d
                    JOIN document_embeddings e ON e.document_id = d.id
                    WHERE d.source = %s
                    ORDER BY e.model
                    """,
                    ("e2e",),
                )
                rows = cur.fetchall()

        models = {model for _, model, _ in rows}
        self.assertIn("nomic-embed-text:v1.5", models)
        self.assertIn("qwen3-embedding:4b", models)

        for _, model, dims in rows:
            self.assertEqual(dims, MODEL_DIMS[model])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
