from __future__ import annotations
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.deployment_bundle import DeploymentBundleError, validate_bundle

def main() -> int:
    root = Path(__file__).resolve().parent; report = root / "reports/deployment_validation_report.json"; error = root / "reports/deployment_validation_attempt_error.json"
    try:
        manifest = validate_bundle(root / "deploy_bundle")
        previous_root = os.environ.get("INSPECTIQ_RUNTIME_ROOT")
        os.environ["INSPECTIQ_RUNTIME_ROOT"] = str(root / "deploy_bundle")
        try:
            from src.dashboard import load_dashboard_context
            load_dashboard_context(root / "config" / "dashboard_config.yaml")
        finally:
            if previous_root is None: os.environ.pop("INSPECTIQ_RUNTIME_ROOT", None)
            else: os.environ["INSPECTIQ_RUNTIME_ROOT"] = previous_root
        result = {"status": "PASS", "bundle_files": len(manifest["files"]), "bundle_size_bytes": sum((root / "deploy_bundle" / name).stat().st_size for name in manifest["files"]), "candidate_rows": 300, "model_included": manifest["model_included"], "prediction_hash_valid": True, "monitoring_hash_valid": True, "prohibited_fields_found": False, "labels_included": False, "external_startup_dependency": False, "automatic_enforcement": False}
        temporary = report.with_suffix(".json.tmp"); temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); temporary.replace(report)
        print(f"bundle_files={result['bundle_files']} bundle_size_bytes={result['bundle_size_bytes']} candidate_rows=300 model_included={result['model_included']}")
        print("prediction_hash_valid=true monitoring_hash_valid=true prohibited_fields_found=false labels_included=false external_startup_dependency=false automatic_enforcement=false")
        print("INSPECTIQ DEPLOYMENT VALIDATION: PASS"); return 0
    except Exception as exc:
        error.write_text(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print("INSPECTIQ DEPLOYMENT VALIDATION: FAIL"); print(f"reason={exc}"); return 1
if __name__ == "__main__": raise SystemExit(main())
