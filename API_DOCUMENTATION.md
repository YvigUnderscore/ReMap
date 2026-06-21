# ReMap API Documentation

Last updated: 2026-04-26

This document covers both API surfaces used by ReMap:

| Surface | Default URL | Audience | Auth |
| --- | --- | --- | --- |
| ReScan/public API | `http://<host>:5000/api/v1` | ReScan iOS app and external HTTP clients | Bearer token, except health |
| Desktop internal API | `http://127.0.0.1:8765/internal/v1` | Local React/Tauri desktop UI | No auth, local use only |

The public API is for uploading ReScan datasets and running remote processing.
The internal API is for the local desktop app: settings, queued jobs, cache,
analytics, artifacts, dependency actions, and server control.

<!-- Image slot: docs/images/remap-api-flow.png
     Suggested: diagram showing ReScan -> public API -> processing job -> output ZIP. -->

## Contents

- [Starting The Servers](#starting-the-servers)
- [Authentication](#authentication)
- [Public ReScan API](#public-rescan-api)
- [Public Processing Settings](#public-processing-settings)
- [Colorspace Values](#colorspace-values)
- [Desktop Internal API](#desktop-internal-api)
- [Desktop Job Payload](#desktop-job-payload)
- [Analytics Payload](#analytics-payload)
- [Dependency Manager](#dependency-manager)
- [Output Bundles](#output-bundles)
- [Error Handling](#error-handling)
- [Curl Workflow](#curl-workflow)
- [Swift Client Notes](#swift-client-notes)

## Starting The Servers

### Public ReScan API

```bash
python remap_server.py
python remap_server.py --host 0.0.0.0 --port 5000 --api-key "your-secret"
python remap_server.py --output-dir /path/to/remap-server-state
```

Command-line options:

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Bind address for remote devices |
| `--port` | `5000` | Public API port |
| `--api-key` | random | Bearer token used by authenticated endpoints |
| `--output-dir` | temp dir | Root directory for uploads, jobs, and output ZIPs |
| `--debug` | off | Enables Flask debug mode |

### Desktop Internal API

```bash
python desktop_backend.py
python desktop_backend.py --host 127.0.0.1 --port 8765
```

The desktop launch scripts start this backend automatically before opening the
Tauri app:

```bash
./launch.sh
```

```bat
launch.bat
```

## Authentication

All public `/api/v1` endpoints except `/health` require:

```http
Authorization: Bearer <API_KEY>
```

The desktop `/internal/v1` API has no authentication and exposes local file
helpers. Keep it bound to `127.0.0.1` unless you are deliberately building a
trusted local-network tool around it.

## Public ReScan API

Base URL:

```text
http://<host>:<port>/api/v1
```

Endpoint summary:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Public health check |
| `POST` | `/upload` | Upload a ReScan ZIP dataset |
| `POST` | `/process` | Start processing a previously uploaded dataset |
| `GET` | `/jobs` | List current in-memory jobs |
| `GET` | `/jobs/{job_id}/status` | Read progress and settings for one job |
| `GET` | `/jobs/{job_id}/logs` | Read job log events |
| `GET` | `/jobs/{job_id}/result` | Download the completed result ZIP |
| `POST` | `/jobs/{job_id}/cancel` | Mark a queued/running public job as cancelled |
| `GET` | `/stats` | Read analytics, telemetry, and throughput |

Public jobs are stored in memory by the standalone server. Restarting
`remap_server.py` clears the public job list.

### `GET /health`

Auth: not required.

Response:

```json
{
  "status": "ok",
  "version": "v1",
  "server": "ReMap",
  "timestamp": "2026-04-26T12:00:00+00:00"
}
```

### `POST /upload`

Auth: required.

Content type: `multipart/form-data`

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | file | yes | ZIP archive containing a ReScan dataset |

Accepted ZIP layout:

```text
dataset.zip
|-- rgb.mp4                  # or rgb.mov
|-- odometry.csv
|-- camera_matrix.csv
|-- depth/                   # optional
`-- confidence/              # optional
```

Image sequence datasets are also accepted:

```text
dataset.zip
|-- rgb/
|   |-- 000001.exr
|   `-- ...
|-- odometry.csv
`-- camera_matrix.csv
```

The dataset may be at the ZIP root or inside a single child folder. ZIP members
with absolute paths or `..` are rejected.

Response `201 Created`:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "files": [
    "rgb.mp4",
    "odometry.csv",
    "camera_matrix.csv"
  ],
  "message": "Dataset uploaded successfully"
}
```

Common errors:

| Status | Reason |
| --- | --- |
| `400` | Missing file, empty filename, invalid ZIP, unsafe ZIP path, or invalid ReScan layout |
| `401` | Missing or malformed Bearer header |
| `403` | Invalid API key |
| `413` | Payload exceeds 10 GB |

### `POST /process`

Auth: required.

Content type: `application/json`

Request:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "input_colorspace": "srgb",
  "output_colorspace": "acescg",
  "settings": {
    "fps": 4.0,
    "feature_type": "superpoint_aachen",
    "matcher_type": "loma_b",
    "max_keypoints": 4096,
    "camera_model": "OPENCV",
    "mapper_type": "GLOMAP",
    "stray_approach": "full_sfm",
    "pairing_mode": "sequential",
    "num_threads": 8,
    "stray_confidence": 2,
    "stray_depth_subsample": 2,
    "stray_gen_pointcloud": true
  }
}
```

Top-level fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | string | yes | ID returned by `/upload` |
| `input_colorspace` | string | no | Source colorspace key; see [Colorspace Values](#colorspace-values) |
| `output_colorspace` | string | no | Output colorspace key; see [Colorspace Values](#colorspace-values) |
| `settings` | object | no | Public processing settings |

Response `202 Accepted`:

```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "queued",
  "message": "Processing started"
}
```

### `GET /jobs`

Auth: required.

Response:

```json
{
  "jobs": [
    {
      "job_id": "f7e8d9c0b1a2",
      "status": "processing",
      "progress": 65,
      "current_step": "Matching",
      "created_at": "2026-04-26T12:00:00+00:00",
      "updated_at": "2026-04-26T12:05:00+00:00"
    }
  ]
}
```

### `GET /jobs/{job_id}/status`

Auth: required.

Response:

```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "processing",
  "progress": 65,
  "current_step": "Matching",
  "created_at": "2026-04-26T12:00:00+00:00",
  "updated_at": "2026-04-26T12:05:00+00:00",
  "error": null,
  "settings": {
    "fps": 4.0,
    "matcher_type": "loma_b"
  }
}
```

Status values:

| Status | Meaning |
| --- | --- |
| `queued` | Job has been created and is waiting |
| `processing` | Pipeline is running |
| `completed` | Result is ready |
| `failed` | Pipeline raised an error |
| `cancelled` | Job was cancelled |

### `GET /jobs/{job_id}/logs`

Auth: required.

Response:

```json
{
  "job_id": "f7e8d9c0b1a2",
  "log": [
    {
      "time": "2026-04-26T12:00:01+00:00",
      "message": "Pipeline started"
    },
    {
      "time": "2026-04-26T12:03:30+00:00",
      "message": "LoMa loma_b: matched 250/900 pairs"
    }
  ]
}
```

### `GET /jobs/{job_id}/result`

Auth: required.

Response `200 OK`:

- `Content-Type: application/zip`
- `Content-Disposition: attachment; filename="remap_result_{job_id}.zip"`

The ZIP contains the output directory produced by the public server, including
`images/`, `hloc_outputs/`, and `sparse/0/`. Recent runs also normalize image
paths inside the final `images.bin` so training tools can resolve bundled image
files reliably.

Errors:

| Status | Reason |
| --- | --- |
| `404` | Job does not exist |
| `409` | Job has not completed yet |
| `500` | Output directory is missing |

### `POST /jobs/{job_id}/cancel`

Auth: required.

Response:

```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "cancelled"
}
```

### `GET /stats`

Auth: required.

Returns the same analytics shape used by the desktop dashboard, adapted from
the public server job list:

```json
{
  "generated_at": "2026-04-26T12:10:00+00:00",
  "system": {
    "cpu_percent": 41.2,
    "ram_percent": 63.5,
    "disk_percent": 72.1
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 5080",
    "capability": "sm_120",
    "memory_used_mb": 9320,
    "memory_total_mb": 16384,
    "temperature_c": 67
  },
  "jobs": {
    "total": 4,
    "active": 1,
    "completed": 2,
    "failed": 1,
    "cancelled": 0
  },
  "usage": {
    "matchers": {
      "superpoint+lightglue": 3,
      "loma_b": 1
    }
  },
  "throughput": {
    "frames_observed": 420,
    "features_observed": 1720320,
    "matches_observed": 184220,
    "avg_feature_ms_per_frame": 44.8,
    "avg_match_ms_per_pair": 19.6
  }
}
```

## Public Processing Settings

All settings are optional. Defaults are applied by `remap_server.py`.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `fps` | number | `4.0` | Target extraction FPS for video/ReScan captures |
| `feature_type` | string | `superpoint_aachen` | `superpoint_aachen`, `superpoint_max`, `disk`, `aliked-n16`, `sift` |
| `matcher_type` | string | `superpoint+lightglue` | `superpoint+lightglue`, `superglue`, `disk+lightglue`, `adalam`, `loma_b`, `loma_g` |
| `max_keypoints` | integer | `4096` | Max features/keypoints per image |
| `camera_model` | string | `OPENCV` | `OPENCV`, `PINHOLE`, `SIMPLE_RADIAL`, `OPENCV_FISHEYE` |
| `mapper_type` | string | `COLMAP` | `COLMAP` or `GLOMAP` |
| `stray_approach` | string | `full_sfm` | `full_sfm` or `known_poses`/ARKit-style labels |
| `pairing_mode` | string | `sequential` | Sequential video pairs or exhaustive pairs |
| `num_threads` | integer | CPU capped | Worker count for conversion and pipeline steps |
| `stray_confidence` | integer | `2` | LiDAR confidence threshold, `0` to `2` |
| `stray_depth_subsample` | integer | `2` | Depth frame subsampling factor |
| `stray_gen_pointcloud` | boolean | `true` | Generate a point cloud from LiDAR depth |
| `label` | string | job ID | Optional human-readable label used by stats |

Recommended starting points:

| Use case | FPS | Feature | Matcher | Mapper |
| --- | --- | --- | --- | --- |
| Normal video | `2-5` | `superpoint_aachen` | `superpoint+lightglue` | `GLOMAP` if available |
| ReScan full SfM | `4-8` | `superpoint_aachen` | `superpoint+lightglue` | `GLOMAP` if available |
| Difficult matching | `2-6` | `superpoint_aachen` | `loma_b` | `COLMAP` or `GLOMAP` |
| Maximum quality sweep | `2-4` | `superpoint_aachen` | `loma_g` | `COLMAP` |

## Colorspace Values

`input_colorspace` and `output_colorspace` accept these case-insensitive keys:

| Value | Canonical processing name | Use |
| --- | --- | --- |
| `linear` | `Linear` | Internal linear light space |
| `srgb` | `sRGB` | Display-referred sRGB |
| `acescg` | `ACES - ACEScg` | ACEScg scene-linear output |
| `aces2065-1` | `ACES2065-1` | ACES interchange/archive |
| `rec709` | `Rec. 709` | Broadcast/video display space |
| `log` | `Log` | Generic logarithmic encoding |
| `raw` | `Raw` | Pass-through interpretation |

The desktop UI has richer built-in labels such as `Apple Log (BT.2020)`,
`HLG (BT.2020)`, `Linear BT.2020`, `ACEScg (EXR + sRGB PNG)`, and
`Custom OCIO...`. Those belong to the internal desktop job payload.

## Desktop Internal API

Base URL:

```text
http://127.0.0.1:8765/internal/v1
```

This API is designed for the local desktop app. It is intentionally broad:
it can expose local file metadata, stream logs, start dependency tasks, and
launch/stop the public API server.

<!-- Image slot: docs/images/remap-internal-api-map.png
     Suggested: diagram showing Tauri UI -> desktop backend -> job service/settings/cache/server controller. -->

Endpoint summary:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Backend health check |
| `GET`, `PUT` | `/settings` | Read or update desktop settings |
| `GET` | `/system/capabilities` | Runtime capabilities plus selectable options |
| `GET` | `/options?ocioPath=...` | Selectable color/features/matchers, optionally from an OCIO file |
| `POST` | `/probe` | Inspect input paths and estimate sampled frames |
| `POST` | `/estimate` | Estimate frames, pairs, disk usage, time, and warnings |
| `GET`, `POST` | `/jobs` | List jobs or create one job |
| `POST` | `/jobs/batch` | Create many jobs at once |
| `DELETE` | `/jobs/queue` | Remove non-processing queued/stopped jobs |
| `GET`, `DELETE` | `/jobs/{job_id}` | Read details or delete one job |
| `GET` | `/jobs/{job_id}/artifacts` | List input/output/media/reconstruction artifacts |
| `GET` | `/jobs/{job_id}/reconstruction` | Sparse preview stats and point sample |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a queued or active job |
| `POST` | `/jobs/{job_id}/pause` | Pause a queued or active job |
| `POST` | `/jobs/{job_id}/resume` | Resume a paused job |
| `GET` | `/jobs/{job_id}/logs/stream` | Server-Sent Events stream for log entries |
| `GET`, `PUT`, `POST` | `/server` | Read/update/start/stop/refresh public API server state |
| `GET` | `/analytics` | Local dashboard telemetry and throughput |
| `GET`, `DELETE` | `/cache` | Read or clear global pipeline cache |
| `GET` | `/dependencies` | Python/package/model cache status |
| `POST` | `/dependencies/actions` | Start a dependency/model maintenance task |
| `GET` | `/files?path=...` | Serve a local file for preview |

### Settings

`GET /settings` returns:

```json
{
  "schema_version": 4,
  "theme": "graphite",
  "defaults": {
    "input_mode": "video",
    "fps_extract": 4,
    "matcher_type": "superpoint+lightglue"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "api_key": "...",
    "auto_start": false,
    "output_dir": ""
  },
  "recent_inputs": [],
  "recent_outputs": [],
  "notifications_enabled": true,
  "system_notifications": true,
  "cache_enabled": true,
  "cache_max_size_gb": 25.0,
  "ram_limit_percent": 90,
  "gpu_vram_limit_percent": 92,
  "blur_threshold": 75.0,
  "black_threshold": 0.08
}
```

Settings are stored in `backend_state/settings.json`, which is ignored by git.

### Capabilities And Options

`GET /system/capabilities` returns runtime detection:

```json
{
  "cpu_count": 16,
  "glomap_available": true,
  "ffmpeg_available": true,
  "ffprobe_available": true,
  "openimageio_available": true,
  "torch_available": true,
  "cuda_available": true,
  "loma_available": true,
  "triton_available": false,
  "torch_device_name": "NVIDIA GeForce RTX 5080",
  "torch_compute_capability": "sm_120",
  "python_version": "3.12.0",
  "platform": "Windows-11",
  "ocio_env": "",
  "features": ["superpoint_aachen", "superpoint_max", "disk", "aliked-n16", "sift"],
  "matchers": ["superpoint+lightglue", "superglue", "disk+lightglue", "adalam", "loma_b", "loma_g"]
}
```

`GET /options` returns only selectable option lists. Add `ocioPath` to load color
spaces from a specific `.ocio` file.

### Probe Inputs

`POST /probe`

```json
{
  "input_mode": "video",
  "input_paths": ["D:/captures/scene.mov"],
  "fps_extract": 4
}
```

Response:

```json
{
  "input_mode": "video",
  "detected_color_profile": "Apple Log (BT.2020)",
  "max_native_fps": 59.94,
  "items": [
    {
      "path": "D:/captures/scene.mov",
      "name": "scene.mov",
      "kind": "video",
      "duration": 120.2,
      "native_fps": 59.94,
      "total_frames": 7200,
      "estimated_frames": 480,
      "color_profile": "Apple Log (BT.2020)",
      "step": 15,
      "valid": true
    }
  ]
}
```

### Estimate Jobs

`POST /estimate`

```json
{
  "requests": [
    {
      "input_mode": "video",
      "input_paths": ["D:/captures/scene.mov"],
      "output_path": "D:/remap/out/scene",
      "fps_extract": 4,
      "matcher_type": "loma_b"
    }
  ]
}
```

Response includes per-job and total frame counts, pair counts, estimated disk
usage, estimated seconds, confidence (`history` or `fallback`), timing
breakdown, and resource warnings.

### Create Jobs

`POST /jobs`

```json
{
  "input_mode": "video",
  "input_paths": ["D:/captures/scene.mov"],
  "output_path": "D:/remap/out/scene",
  "label": "Kitchen walk",
  "fps_extract": 4,
  "matcher_type": "superpoint+lightglue",
  "mapper_type": "GLOMAP"
}
```

Response `202 Accepted` returns a `JobDetail`.

`POST /jobs/batch`

```json
{
  "requests": [
    { "input_paths": ["D:/captures/a.mov"], "output_path": "D:/out/a" },
    { "input_paths": ["D:/captures/b.mov"], "output_path": "D:/out/b" }
  ]
}
```

The job service persists job metadata in `backend_state/jobs.json`. Jobs that
were `processing` during a backend shutdown are marked `interrupted` on restart.

### Job Details

`GET /jobs/{job_id}` returns:

```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "processing",
  "progress": 42,
  "current_step": "Feature extraction",
  "label": "Kitchen walk",
  "output_path": "D:/remap/out/scene",
  "input_mode": "video",
  "queue_position": 1,
  "progress_note": "ETA 2m 14s",
  "eta_seconds": 134,
  "request": { "...": "..." },
  "logs": [
    {
      "id": 12,
      "timestamp": "2026-04-26T12:02:00+00:00",
      "level": "info",
      "message": "Feature extraction started"
    }
  ]
}
```

### Live Log Stream

`GET /jobs/{job_id}/logs/stream`

Server-Sent Events format:

```text
id: 12
event: log
data: {"id":12,"timestamp":"...","level":"info","message":"..."}
```

The stream stops when the job reaches a terminal state.

### Artifacts And Reconstruction

`GET /jobs/{job_id}/artifacts` returns grouped files:

| Field | Meaning |
| --- | --- |
| `input_paths` | Source files/folders, with samples for folders |
| `output_path` | Output root |
| `frame_dirs` | Candidate frame directories |
| `frames` | Previewable non-EXR frames |
| `exrs` | EXR outputs |
| `reconstruction` | `.bin`, `.txt`, `.ply`, `.json`, `.nvm`, `.glb`, `.obj` files |
| `latest_outputs` | Recent output files recursively collected from the output root |

`GET /jobs/{job_id}/reconstruction` returns sparse preview data:

```json
{
  "available": true,
  "sparse_dir": "D:/remap/out/scene/scene_SfM_Dataset_Output",
  "stats": {
    "camera_count": 1,
    "image_count": 420,
    "registered_images": 408,
    "point_count": 184233,
    "mean_reprojection_error": 0.71
  },
  "points": {
    "count": 184233,
    "sample": [
      { "id": 1, "xyz": [0.1, 0.2, 1.3], "rgb": [180, 172, 160], "error": 0.42 }
    ]
  }
}
```

### Public Server Control

`GET /server` returns current public server state:

```json
{
  "config": {
    "host": "0.0.0.0",
    "port": 5000,
    "api_key": "...",
    "auto_start": false,
    "output_dir": ""
  },
  "running": true,
  "local_ip": "192.168.1.42",
  "connect_url": "http://192.168.1.42:5000",
  "health": {
    "reachable": true,
    "url": "http://127.0.0.1:5000/api/v1/health"
  },
  "remote_jobs": []
}
```

`PUT /server` updates `host`, `port`, `api_key`, `auto_start`, or
`output_dir`.

`POST /server` accepts:

```json
{ "action": "start" }
```

Actions: `start`, `stop`, or `refresh`.

### Cache

`GET /cache` reports the global feature/pair/match cache in
`backend_state/cache`.

`DELETE /cache` removes all cache entries.

The cache currently stores reusable HLoc/LoMa artifacts such as `features.h5`,
`pairs.txt`, `matches.h5`, `loma_features.h5`, and `loma_matches.h5`.

## Desktop Job Payload

The desktop job request is richer than the public server settings because it
drives local UI workflows, cache, quality checks, and output normalization.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `input_mode` | string | `video` | `video`, `images`, or `rescan` |
| `input_paths` | string[] | `[]` | Files or folders selected by the user |
| `output_path` | string | `""` | Required output root |
| `fps_extract` | number | `4` | Target extraction FPS |
| `force_16bit` | boolean | `false` | Legacy 16-bit output flag |
| `camera_model` | string | `OPENCV` | COLMAP camera model |
| `feature_type` | string | `superpoint_aachen` | Feature extractor |
| `matcher_type` | string | `superpoint+lightglue` | Matcher, including `loma_b` and `loma_g` |
| `max_keypoints` | integer | `4096` | Keypoint limit |
| `pairing_mode` | string | `Sequential (Video)` | UI pairing label |
| `mapper_type` | string | detected | `GLOMAP` when available, otherwise `COLMAP` |
| `stray_approach` | string | `full_sfm` | ReScan full SfM or known poses |
| `stray_confidence` | integer | `2` | ReScan LiDAR confidence threshold |
| `stray_depth_subsample` | integer | `2` | ReScan depth subsampling |
| `stray_gen_pointcloud` | boolean | `true` | Generate LiDAR point cloud |
| `color_enabled` | boolean | `false` | Enable color conversion |
| `color_source` | string | `Auto-detect` | Built-in source label or custom OCIO |
| `color_dest` | string | `ACEScg (EXR + sRGB PNG)` | Built-in output label or custom OCIO |
| `detected_color_profile` | string | `""` | Filled from probing when available |
| `ocio_path` | string | `""` | OCIO config path |
| `ocio_in_cs` | string | `""` | Custom OCIO input colorspace |
| `ocio_out_cs` | string | `""` | Custom OCIO output colorspace |
| `keep_srgb_png` | boolean | `true` | Keep preview PNGs next to EXR outputs |
| `use_acescg_exr` | boolean | `true` | Prefer EXR final bundle when color pipeline requests it |
| `num_workers` | integer | CPU count | Worker count |
| `server_port` | integer | `5000` | Public server default port stored in job defaults |
| `server_api_key` | string | random | Public server key stored in job defaults |
| `label` | string | `""` | Human-readable job name |
| `skip_existing` | boolean | `true` | Reuse existing outputs/cache where possible |
| `quality_sweep` | boolean | `false` | Mark quality-sweep jobs |
| `sweep_sample_frames` | integer | `80` | Sample target for quality sweeps |
| `exclude_blurry` | boolean | `false` | Move blurry frames to `_rejected_frames` |
| `exclude_black` | boolean | `false` | Move near-black frames to `_rejected_frames` |
| `blur_threshold` | number | `75.0` | Laplacian blur threshold |
| `black_threshold` | number | `0.08` | Mean luma threshold |
| `ram_limit_percent` | integer | `90` | Warning threshold for estimates |
| `gpu_vram_limit_percent` | integer | `92` | Warning threshold for estimates |

## Analytics Payload

`GET /analytics` and public `GET /stats` share this broad shape:

| Field | Description |
| --- | --- |
| `generated_at` | UTC generation time |
| `system` | CPU, RAM, process RSS, disk usage |
| `gpu` | Torch CUDA and `nvidia-smi` snapshot when available |
| `maxima` | Peak CPU/RAM/GPU values seen during the backend session |
| `jobs` | Counts and recent job rows |
| `usage` | Counts by matcher, feature, and input mode |
| `throughput` | Observed frames, features, matches, pairs, and average timings |

Recent job rows include matcher, feature, input mode, frame count, feature and
match counts, point count, output size, duration, step timings, and per-frame or
per-pair timing where enough data exists.

## Dependency Manager

`GET /dependencies` returns:

```json
{
  "python": ".venv/Scripts/python.exe",
  "requirements": "requirements.txt",
  "requirements_lock": "requirements.lock.txt",
  "loma_locked_url": "git+https://github.com/davnords/LoMa.git@9105854833f55d18194d0505d913f0a74b194ef0#egg=lomatch",
  "packages": [
    {
      "name": "lightglue",
      "installed": true,
      "version": "0.0",
      "direct_url": "{...}"
    }
  ],
  "models": {
    "torch_hub_dir": "...",
    "torch_hub_checkpoints": [],
    "superglue_weights": [],
    "total_size_mb": 142.4
  },
  "task": {
    "running": false,
    "action": "",
    "status": "idle",
    "message": "",
    "log": []
  }
}
```

`POST /dependencies/actions`

```json
{ "action": "download_all_models" }
```

Supported actions:

| Action | Description |
| --- | --- |
| `install_packages` | Reinstall the stable pinned baseline and LoMa workaround |
| `update_packages` | Update toward the same pinned baseline |
| `download_core_models` | Warm HLoc, LightGlue, and SuperGlue caches |
| `download_loma_b` | Warm LoMa-B cache |
| `download_loma_g` | Warm LoMa-G cache |
| `download_all_models` | Warm all core and LoMa model caches |

Only one dependency task can run at a time. A second task returns `409`.

## Output Bundles

Desktop jobs normalize the final trainer bundle to:

```text
<output_path>/
|-- <output_name>_SfM_Dataset_Output/
|   |-- cameras.bin
|   |-- images.bin
|   |-- points3D.bin
|   |-- images/
|   `-- images_srgb_png/        # optional for EXR workflows
|-- hloc_outputs/
|-- sparse/0/
`-- .remap/
```

Public API result ZIPs include the standalone server output directory. The
latest path normalization keeps bundled image references relative to the final
model folder where possible.

## Error Handling

Errors follow a consistent JSON shape:

```json
{
  "error": "Invalid Request",
  "message": "Missing output_path"
}
```

Common HTTP codes:

| Status | Meaning |
| --- | --- |
| `200` | Request succeeded |
| `201` | Upload created |
| `202` | Processing/dependency task accepted |
| `400` | Invalid request payload |
| `401` | Missing Bearer token on public API |
| `403` | Invalid public API token |
| `404` | Unknown dataset/job/file |
| `409` | Job/task is not in a valid state for the requested action |
| `413` | Public upload is larger than 10 GB |
| `500` | Server-side failure |

## Curl Workflow

```bash
# 1. Health check
curl http://localhost:5000/api/v1/health

# 2. Upload a ReScan dataset
curl -X POST http://localhost:5000/api/v1/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@/path/to/rescan_dataset.zip"

# 3. Start processing
curl -X POST http://localhost:5000/api/v1/process \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "a1b2c3d4e5f6",
    "input_colorspace": "srgb",
    "output_colorspace": "acescg",
    "settings": {
      "fps": 4.0,
      "feature_type": "superpoint_aachen",
      "matcher_type": "superpoint+lightglue",
      "mapper_type": "GLOMAP",
      "stray_approach": "full_sfm"
    }
  }'

# 4. Poll status
curl http://localhost:5000/api/v1/jobs/f7e8d9c0b1a2/status \
  -H "Authorization: Bearer YOUR_API_KEY"

# 5. Read logs
curl http://localhost:5000/api/v1/jobs/f7e8d9c0b1a2/logs \
  -H "Authorization: Bearer YOUR_API_KEY"

# 6. Download result after status is completed
curl -O -J http://localhost:5000/api/v1/jobs/f7e8d9c0b1a2/result \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Swift Client Notes

Use the public API from ReScan or any Swift client with normal `URLSession`
requests:

```swift
var request = URLRequest(url: URL(string: "\(baseURL)/api/v1/process")!)
request.httpMethod = "POST"
request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONSerialization.data(withJSONObject: [
    "dataset_id": datasetId,
    "input_colorspace": "srgb",
    "output_colorspace": "acescg",
    "settings": [
        "fps": 4.0,
        "feature_type": "superpoint_aachen",
        "matcher_type": "superpoint+lightglue",
        "mapper_type": "GLOMAP",
        "stray_approach": "full_sfm"
    ]
])
```

For upload, send `multipart/form-data` with the ZIP file under field name
`file`. For progress, poll `/jobs/{job_id}/status` every few seconds and stop
when the status is `completed`, `failed`, or `cancelled`.
