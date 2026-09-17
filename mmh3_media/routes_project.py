"""Thin ComfyUI HTTP adapter; blocking archive work stays off the event loop."""
import asyncio
from aiohttp import web

from .errors import MMH3Error
from .project_actions import ProjectPublicationBusy, ProjectStateConflict
from .project_manager import ProjectManager
from .archive import load_archive, materialize_resource_file
from .archive_preview import read_archive_preview
from .project_storage import storage_report, move_candidate
from .project_timeline import project_timeline, assembly_preview, export_project, export_path


def register_project_routes(routes, input_root, output_root):
    manager = ProjectManager(input_root, output_root)
    slots = asyncio.Semaphore(2)

    async def call(fn, *args, **kwargs):
        async with slots:
            return await asyncio.to_thread(fn, *args, **kwargs)

    @routes.get("/mmh3_media/projects")
    async def projects(request):
        return web.json_response(await call(manager.list_projects), headers={"Cache-Control": "no-store"})

    @routes.post("/mmh3_media/project")
    async def action(request):
        # No cross-origin mutations, including form POSTs. Same-origin fetch sends JSON.
        origin = request.headers.get("Origin")
        if (origin and origin != f"{request.scheme}://{request.host}") or request.content_type != "application/json":
            return web.json_response({"error": "Same-origin JSON request required"}, status=403)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            operation, project_id = body.get("action"), body.get("project_id")
            expected = body.get("expected_state_digest")
            if operation == "create":
                result = await call(manager.create, body.get("file"))
            elif operation == "state":
                result = await call(manager.state, project_id)
            elif operation == "add":
                result = await call(manager.add_candidate, project_id, body.get("file"), expected)
            elif operation in {"select", "reject"}:
                result = await call(manager.choose, project_id, body.get("candidate_id"), expected, operation)
            elif operation == "impact":
                result = await call(manager.impact, project_id, body.get("candidate_id"), expected)
            elif operation == "accept":
                result = await call(manager.accept, project_id, body.get("candidate_id"), expected, body.get("operation_id"))
            elif operation == "handoff":
                result = await call(manager.handoff, project_id, expected, body.get("target_segment_id", ""),
                    body.get("parent_file", ""), body.get("prompts", ""), body.get("seed", 0))
            elif operation == "branch_preview":
                result = await call(manager.branch_preview, project_id, expected,
                    body.get("candidate_id", ""), body.get("source_file", ""))
            elif operation == "branch":
                result = await call(manager.branch, project_id, expected, body.get("source_sha256"),
                    body.get("operation_id"), body.get("candidate_id", ""), body.get("source_file", ""), body.get("name", ""))
            elif operation == "storage":
                result = await call(storage_report, manager, project_id)
            elif operation == "timeline":
                result = await call(project_timeline, manager, project_id)
            elif operation == "attach_segment":
                result = await call(manager.attach_segment, project_id, body.get("file"), expected)
            elif operation == "assembly_preview":
                result = await call(assembly_preview, manager, project_id, expected)
            elif operation == "export":
                result = await call(export_project, manager, project_id, expected, body.get("assembly_digest"))
            elif operation in {"trash", "restore"}:
                result = await call(move_candidate, manager, project_id, body.get("candidate_id"), expected,
                    body.get("storage_digest"), restore=operation == "restore")
            else:
                raise ValueError("Unknown project action")
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ProjectStateConflict, ProjectPublicationBusy) as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (MMH3Error, ValueError, TypeError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError:
            return web.json_response({"error": "Project archive could not be read or written"}, status=400)

    @routes.get("/mmh3_media/project_media")
    async def media(request):
        try:
            project_id, candidate_id = request.query.get("project_id"), request.query.get("candidate_id")
            path = await call(manager.candidate, project_id, candidate_id) if candidate_id else manager.current(project_id)
            if request.query.get("kind") == "video":
                def video():
                    packet = load_archive(path, verify="on_access")
                    resource = packet.get_primary("video")
                    if resource is None:
                        raise ValueError("No decoded video preview in this archive")
                    return materialize_resource_file(packet, resource)
                return web.FileResponse(await call(video), headers={"Cache-Control": "no-store"})
            data, mime, _ = await call(read_archive_preview, path)
            return web.Response(body=data, content_type=mime, headers={"Cache-Control": "no-store"})
        except (MMH3Error, ValueError, OSError):
            return web.json_response({"error": "Preview unavailable; no generation or decoding was performed"}, status=404)

    @routes.get("/mmh3_media/project_export")
    async def download(request):
        try:
            path = await call(export_path, manager, request.query.get("project_id"), request.query.get("export_id"))
            return web.FileResponse(path, headers={"Content-Disposition": 'attachment; filename="final.mp4"', "Cache-Control": "no-store"})
        except (MMH3Error, ValueError, OSError):
            return web.json_response({"error": "Export unavailable"}, status=404)
