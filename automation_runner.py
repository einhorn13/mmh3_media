from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from mmh3_media.automation_execution import build_execution_summary, load_execution_ledger, save_execution_summary
from mmh3_media.automation_runner import ComfyQueueClient, format_execution_summary, run_execution_ledger, run_final_assembly
from mmh3_media.runner_control import read_runner_control, write_runner_control


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an MMH3 execution ledger sequentially through a ComfyUI API workflow.")
    parser.add_argument("--workflow", help="Canonical API workflow JSON containing MMH3 placeholders.")
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
    parser.add_argument("--output-root", help="Local ComfyUI output directory for receipt-verified artifact recovery when history was cleared.")
    parser.add_argument("--control", help="Control JSON path; defaults to <checkpoint>.control.json.")
    parser.add_argument("--request", choices=["pause", "cancel", "resume"],
                        help="Send a control request without starting another runner. Pause drains the current segment.")
    args = parser.parse_args()

    ledger_path = Path(args.ledger).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else ledger_path
    control_path = Path(args.control).resolve() if args.control else checkpoint_path.with_name(checkpoint_path.name + ".control.json")
    if args.request:
        write_runner_control(control_path, "run" if args.request == "resume" else args.request)
        print(f"CONTROL {args.request} -> {control_path}")
        return 0
    if not args.workflow:
        parser.error("--workflow is required when starting a runner")
    workflow = json.loads(Path(args.workflow).read_text(encoding="utf-8"))
    ledger = load_execution_ledger(checkpoint_path if checkpoint_path.exists() else ledger_path)
    client = ComfyQueueClient(args.server)
    interrupts = 0
    def request_stop(_signum, _frame):
        nonlocal interrupts
        interrupts += 1
        action = "pause" if interrupts == 1 else "cancel"
        write_runner_control(control_path, action)
        print("PAUSE requested: finish current segment. Press Ctrl+C again to cancel it." if action == "pause"
              else "CANCEL requested for this runner's prompt only.", flush=True)
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    if args.continue_on_error:
        continue_on_error = True
    elif args.error_policy == "continue_on_error":
        continue_on_error = True
    elif args.error_policy == "stop_on_error":
        continue_on_error = False
    else:
        continue_on_error = ledger.get("error_policy") == "continue_on_error"
    try:
        final = run_execution_ledger(
            ledger,
            workflow,
            checkpoint_path=checkpoint_path,
            client=client,
            save_node_id=args.save_node_id,
            resume_mode=args.resume_mode,
            continue_on_error=continue_on_error,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.job_timeout,
            control=lambda: read_runner_control(control_path),
            output_root=args.output_root,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    summary = build_execution_summary(final)
    summary_path = Path(args.summary).resolve() if args.summary else checkpoint_path.with_name(checkpoint_path.name + ".summary.json")
    save_execution_summary(summary, summary_path)
    print(f"FINAL status={final['status']} revision={final['revision']}")
    print(format_execution_summary(summary))
    print(f"REPORT {summary_path}")
    if final["status"] == "completed" and args.assembly_workflow and summary["assembly"]["ready"] and read_runner_control(control_path) == "run":
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
    if (read_runner_control(control_path) == "pause"
            and not any(job["state"] in {"running", "failed"} for job in final["jobs"])):
        return 0
    return 0 if final["status"] == "completed" and not any(issue.get("severity") == "error" for issue in summary["issues"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
