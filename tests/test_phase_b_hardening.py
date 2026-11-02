"""
Acceptance Test Suite for Phase B Hardening (P1-16, P2-02, P2-03).
Covers:
- P1-16: Canonical port mapping harmonization across nginx, scripts, and documentation
- P2-02: Dashboard API Authentication (X-API-Key, Bearer, query param), RBAC (admin vs viewer), and Origin-Restricted CORS
- P2-03: Dependency manifests complete with optional extras ([ml], [dashboard], [research], [dev], [all])
"""
import os
import re
import tomllib
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from gold_trader.api.dashboard_api import app, verify_admin_access, verify_read_access


class TestPhaseBPortHarmonization(unittest.TestCase):
    """Verifies P1-16 port mapping across all configuration and launch artifacts."""

    def setUp(self):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_nginx_conf_proxies_canonical_ports(self):
        nginx_path = os.path.join(self.repo_root, "scripts", "nginx_trading_dashboard.conf")
        self.assertTrue(os.path.exists(nginx_path))
        with open(nginx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # UI must proxy to port 5000 (Streamlit production port)
        self.assertIn("proxy_pass http://127.0.0.1:5000;", content)
        # API must proxy to port 8787 (FastAPI production port)
        self.assertIn("proxy_pass http://127.0.0.1:8787/api/;", content)
        # Stale default ports 8501 or 8000 must NOT be active proxy_pass targets
        self.assertNotIn("proxy_pass http://127.0.0.1:8501;", content)
        self.assertNotIn("proxy_pass http://127.0.0.1:8000/api/;", content)

    def test_run_scripts_use_canonical_ports(self):
        ui_script = os.path.join(self.repo_root, "scripts", "run_dashboard_ui.sh")
        api_script = os.path.join(self.repo_root, "scripts", "run_dashboard_api.sh")

        with open(ui_script, "r", encoding="utf-8") as f:
            ui_content = f.read()
        with open(api_script, "r", encoding="utf-8") as f:
            api_content = f.read()

        self.assertIn("--server.port 5000", ui_content)
        self.assertIn("http://127.0.0.1:8787", ui_content)
        self.assertIn("--server.enableXsrfProtection true", ui_content)
        self.assertIn("--port 8787", api_content)

    def test_readme_documents_canonical_ports(self):
        readme_path = os.path.join(self.repo_root, "atlas_trading_dashboard", "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        self.assertIn("8787", readme_content)
        self.assertIn("5000", readme_content)
        self.assertIn("http://127.0.0.1:8787", readme_content)


class TestPhaseBDashboardAPIAuthAndCORS(unittest.TestCase):
    """Verifies P2-02 API Key Auth, Admin Privileges, and CORS restrictions."""

    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        # Reset auth environment variables
        for var in ("AEGIS_API_KEY", "AEGIS_ADMIN_KEY", "AEGIS_VIEWER_KEY", "AEGIS_CORS_ORIGINS"):
            if var in os.environ:
                del os.environ[var]

    def test_health_endpoint_is_always_public(self):
        os.environ["AEGIS_API_KEY"] = "super-secret-key"
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("components", data)

    def test_unauthenticated_requests_rejected_when_auth_enabled(self):
        os.environ["AEGIS_API_KEY"] = "production-token-123"

        # Overview endpoint without auth must return 401
        res = self.client.get("/api/dashboard/overview")
        self.assertEqual(res.status_code, 401)
        self.assertIn("Unauthorized", res.json().get("detail", ""))

        # Journal list without auth must return 401
        res = self.client.get("/api/journal/list")
        self.assertEqual(res.status_code, 401)

    def test_authenticated_requests_succeed_with_header_bearer_or_query(self):
        os.environ["AEGIS_API_KEY"] = "production-token-123"

        with patch("gold_trader.api.dashboard_api.fetch_historic_and_pnl_data") as mock_fetch:
            mock_fetch.return_value = {"treasury_model_2": {"balance": 1500.0, "swept": 0.0}}

            # 1. Header: X-API-Key
            res1 = self.client.get("/api/dashboard/treasury", headers={"X-API-Key": "production-token-123"})
            self.assertEqual(res1.status_code, 200)

            # 2. Header: Authorization Bearer
            res2 = self.client.get("/api/dashboard/treasury", headers={"Authorization": "Bearer production-token-123"})
            self.assertEqual(res2.status_code, 200)

            # 3. Query Parameter: ?api_key=...
            res3 = self.client.get("/api/dashboard/treasury?api_key=production-token-123")
            self.assertEqual(res3.status_code, 200)

    def test_admin_vs_viewer_role_authorization_on_mutation_endpoints(self):
        os.environ["AEGIS_API_KEY"] = "admin-secret-999"
        os.environ["AEGIS_VIEWER_KEY"] = "read-only-viewer-111"

        with patch("gold_trader.core.daily_journal.DailyJournalEngine.generate_journal") as mock_gen, \
             patch("gold_trader.core.daily_journal.DailyJournalEngine.backfill_all") as mock_bf:
            mock_gen.return_value = "/tmp/journal.md"
            mock_bf.return_value = ["/tmp/j1.md"]

            # Viewer key cannot trigger POST /api/journal/generate (Must return 403 Forbidden)
            res_viewer_gen = self.client.post("/api/journal/generate", headers={"X-API-Key": "read-only-viewer-111"})
            self.assertEqual(res_viewer_gen.status_code, 403)
            self.assertIn("Admin privileges required", res_viewer_gen.json().get("detail", ""))

            # Viewer key cannot trigger POST /api/journal/backfill (Must return 403 Forbidden)
            res_viewer_bf = self.client.post("/api/journal/backfill", headers={"X-API-Key": "read-only-viewer-111"})
            self.assertEqual(res_viewer_bf.status_code, 403)

            # Admin key succeeds
            res_admin_gen = self.client.post("/api/journal/generate", headers={"X-API-Key": "admin-secret-999"})
            self.assertEqual(res_admin_gen.status_code, 200)

            res_admin_bf = self.client.post("/api/journal/backfill", headers={"X-API-Key": "admin-secret-999"})
            self.assertEqual(res_admin_bf.status_code, 200)

    def test_origin_restricted_cors_headers(self):
        # Trusted origin gets Access-Control-Allow-Origin header
        res_trusted = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5000",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(res_trusted.headers.get("access-control-allow-origin"), "http://localhost:5000")

        # Untrusted origin does not receive Access-Control-Allow-Origin header
        res_untrusted = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://malicious-attacker-site.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotEqual(res_untrusted.headers.get("access-control-allow-origin"), "http://malicious-attacker-site.com")


class TestPhaseBDependencyManifests(unittest.TestCase):
    """Verifies P2-03 pyproject.toml and requirements.txt completeness."""

    def setUp(self):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_pyproject_toml_extras_and_core_declared(self):
        pyproject_path = os.path.join(self.repo_root, "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            cfg = tomllib.load(f)

        project = cfg.get("project", {})
        dependencies = project.get("dependencies", [])
        dep_str = " ".join(dependencies)

        # Core dependencies
        self.assertIn("numpy", dep_str)
        self.assertIn("pandas", dep_str)
        self.assertIn("pydantic", dep_str)
        self.assertIn("requests", dep_str)
        self.assertIn("yfinance", dep_str)

        # Optional dependencies / extras
        extras = project.get("optional-dependencies", {})
        self.assertIn("ml", extras)
        self.assertIn("dashboard", extras)
        self.assertIn("research", extras)
        self.assertIn("dev", extras)
        self.assertIn("all", extras)

        # Verify ml extra contains scikit-learn and lightgbm
        ml_str = " ".join(extras["ml"])
        self.assertIn("scikit-learn", ml_str)
        self.assertIn("lightgbm", ml_str)

        # Verify dashboard extra contains fastapi, uvicorn, streamlit, plotly
        dash_str = " ".join(extras["dashboard"])
        self.assertIn("fastapi", dash_str)
        self.assertIn("uvicorn", dash_str)
        self.assertIn("streamlit", dash_str)
        self.assertIn("plotly", dash_str)

        # Verify research extra contains openbb
        research_str = " ".join(extras["research"])
        self.assertIn("openbb", research_str)

    def test_requirements_txt_matches_declared_components(self):
        req_path = os.path.join(self.repo_root, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            req_content = f.read()

        for pkg in ["numpy", "pandas", "pydantic", "lightgbm", "scikit-learn", "fastapi", "uvicorn", "streamlit", "plotly", "openbb"]:
            self.assertIn(pkg, req_content)


if __name__ == "__main__":
    unittest.main()
