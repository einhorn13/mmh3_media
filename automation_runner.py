from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmh3_media.automation_execution import build_execution_summary, load_execution_ledger, save_execution_summary
from mmh3_media.automation_runner import ComfyQueueClient, format_execution_summary, run_execution_ledger, run_final_assembly


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an MMH3 execution ledger sequentially through a ComfyUI API workflow.")
    parser.add_argument("--workflow", required=True, help="Canonical API workflow JSON containing MMH3 placeholders.")
    parser.add_argument("--ledger", required=True, help="Existing execution ledger JSON.")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to --ledger.")
    parser.add_argument("--summary", help="Post-run summary JSON path; defaults to <checkpoint>.summary.json.")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--save-node-id", default="")
    parser.add_argument("--assembly-workflow", help="Optional final assembly API workflow containing __MMH3_LEDGER_JSON__.")
    parser.add_argument("--assembly-save-node-id", default="")
    parser.add_argument("--resume-mode", choices=["pending_and_failed", "failed_only", "include_cancelled"], default="pending_and_failed")
    parser.add_argument("--continue-on-error", action="store_true", help="Override the ledger and continue after a failed job.")
    parser.add_argument("--error-policy", choices=["ledger", "stop_on_error", "continue_on_error"], default="ledger", help="Execution failure policy; default uses the immutable ledger selection.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--job-timeout", type=float, default=86400.0)
    args = parser.parse_args()

    ledger_path = Path(args.ledger).resolve()
    workflow = json.loads(Path(args.workflow).read_text(encoding="utf-8"))
    ledger = load_execution_ledger(ledger_path)
    client = ComfyQueueClient(args.server)
    if args.continue_on_error:
        continue_on_error = True
    elif args.error_policy == "continue_on_error":
        continue_on_error = True
    elif args.error_policy == "stop_on_error":
        continue_on_error = False
    else:
        continue_on_error = ledger.get("error_policy") == "continue_on_error"
    final = run_execution_ledger(
        ledger,
        workflow,
        checkpoint_path=Path(args.checkpoint).resolve() if args.checkpoint else ledger_path,
        client=client,
        save_node_id=args.save_node_id,
        resume_mode=args.resume_mode,
        continue_on_error=continue_on_error,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.job_timeout,
    )
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else ledger_path
    summary = build_execution_summary(final)
    summary_path = Path(args.summary).resolve() if args.summary else checkpoint_path.with_name(checkpoint_path.name + ".summary.json")
    save_execution_summary(summary, summary_path)
    print(f"FINAL status={final['status']} revision={final['revision']}")
    print(format_execution_summary(summary))
    print(f"REPORT {summary_path}")
    if final["status"] == "completed" and args.assembly_workflow and summary["assembly"]["ready"]:
        assembly = json.loads(Path(args.assembly_workflow).read_text(encoding="utf-8"))
        final_path = run_final_assembly(
            final,
            assembly,
            client=client,
            save_node_id=args.assembly_save_node_id,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.job_timeout,
        )
        print(f"ASSEMBLED {final_path}")
    return 0 if final["status"] == "completed" and not any(issue.get("severity") == "error" for issue in summary["issues"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
