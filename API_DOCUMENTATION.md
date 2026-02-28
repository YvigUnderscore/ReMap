# ReMap API Documentation

Complete REST API reference for the **ReMap Server**, designed for integration with the **ReScan** iOS app and any HTTP client.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Upload Dataset](#upload-dataset)
  - [Start Processing](#start-processing)
  - [Job Status](#job-status)
  - [Job Logs](#job-logs)
  - [Job Result (Download)](#job-result-download)
  - [List Jobs](#list-jobs)
  - [Cancel Job](#cancel-job)
- [Processing Settings](#processing-settings)
- [Colorspace Values](#colorspace-values)
- [Error Handling](#error-handling)
- [Workflow Example](#workflow-example)
- [iOS / Swift Integration Guide](#ios--swift-integration-guide)

---

## Overview

The ReMap API server allows the ReScan iOS app (or any authenticated client) to:

1. **Upload** a ReScan dataset (ZIP archive containing LiDAR capture data)
2. **Start** the full SfM processing pipeline with custom settings
3. **Monitor** job progress in real time
4. **Download** the final COLMAP result when processing is complete

```
┌──────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  📱 ReScan   │  HTTP   │  🖥️ ReMap Server │  output │  📁 COLMAP      │
│  (iOS App)   │ ──────► │  (REST API)      │ ──────► │  Dataset (ZIP)  │
│              │         │                  │         │                 │
│ POST /upload │         │ Process pipeline │         │ images/         │
│ POST /process│         │ • Stray→COLMAP   │         │ sparse/0/       │
│ GET  /status │         │ • Features       │         │   cameras.bin   │
│ GET  /result │         │ • Matching       │         │   images.bin    │
│              │         │ • SfM            │         │   points3D.bin  │
└──────────────┘         └──────────────────┘         └─────────────────┘
```

**Base URL**: `http://<host>:<port>/api/v1`

**Default port**: `5000`

---

## Getting Started

### Method 1: Standalone Server

```bash
# Start the server (API key is auto-generated)
python remap_server.py

# Start with a custom port and API key
python remap_server.py --port 8080 --api-key "my-secret-key"

# Start with a custom output directory
python remap_server.py --output-dir /path/to/output
```

### Method 2: From the ReMap GUI

1. Open ReMap GUI (`python ReMap-GUI.py`)
2. Scroll to the **🌐 API Server (ReScan Remote)** section
3. Set the desired port (default: 5000)
4. Click **▶ Start Server**
5. Copy the generated API key displayed in the field

### Command-Line Options

| Option         | Default     | Description                              |
|----------------|-------------|------------------------------------------|
| `--host`       | `0.0.0.0`  | Bind address                             |
| `--port`       | `5000`     | Port number                              |
| `--api-key`    | *(random)* | API key for authentication               |
| `--output-dir` | *(temp)*   | Root directory for job outputs           |
| `--debug`      | `false`    | Enable Flask debug mode                  |

---

## Authentication

All endpoints **except** `/api/v1/health` require a **Bearer token** in the `Authorization` header.

```
Authorization: Bearer <API_KEY>
```

The API key is generated when the server starts (printed to the console or displayed in the GUI).

### Error Responses

| Code | Scenario                       |
|------|-------------------------------|
| 401  | Missing Authorization header   |
| 403  | Invalid API key                |

---

## Endpoints

### Health Check

Check if the server is running.

```
GET /api/v1/health
```

**Authentication**: Not required

**Response** `200 OK`:
```json
{
  "status": "ok",
  "version": "v1",
  "server": "ReMap",
  "timestamp": "2025-01-15T10:30:00.000000+00:00"
}
```

---

### Upload Dataset

Upload a ReScan dataset as a ZIP archive.

```
POST /api/v1/upload
```

**Authentication**: Required

**Content-Type**: `multipart/form-data`

**Form fields**:

| Field  | Type   | Required | Description                             |
|--------|--------|----------|-----------------------------------------|
| `file` | File   | Yes      | ZIP archive containing the ReScan dataset |

**ZIP contents** (required files):

| File                  | Description                            |
|-----------------------|----------------------------------------|
| `rgb.mp4` or `rgb.mov` | Apple ProRes Log video                |
| `odometry.csv`        | ARKit camera poses (position + quaternion) |
| `camera_matrix.csv`   | Camera intrinsics (3×3 matrix)         |

Optional files: `depth/*.png` (LiDAR depth maps), `confidence/*.png`

**Response** `201 Created`:
```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "files": [
    "rgb.mp4",
    "odometry.csv",
    "camera_matrix.csv",
    "depth/000001.png",
    "depth/000002.png"
  ],
  "message": "Dataset uploaded successfully"
}
```

**Error responses**:

| Code | Reason                                       |
|------|----------------------------------------------|
| 400  | No file attached, invalid ZIP, or missing required ReScan files |
| 401  | Missing authorization                        |
| 413  | File too large (max 10 GB)                   |

---

### Start Processing

Start the SfM pipeline on a previously uploaded dataset.

```
POST /api/v1/process
```

**Authentication**: Required

**Content-Type**: `application/json`

**Request body**:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "input_colorspace": "srgb",
  "output_colorspace": "acescg",
  "settings": {
    "fps": 4.0,
    "feature_type": "superpoint_aachen",
    "matcher_type": "superpoint+lightglue",
    "max_keypoints": 4096,
    "camera_model": "OPENCV",
    "mapper_type": "COLMAP",
    "stray_approach": "full_sfm",
    "pairing_mode": "sequential",
    "num_threads": 8,
    "color_pipeline": "None",
    "stray_confidence": 2,
    "stray_depth_subsample": 2,
    "stray_gen_pointcloud": true
  }
}
```

All fields in `settings` are **optional** — defaults are applied automatically. See [Processing Settings](#processing-settings) for details.

**Top-level fields**:

| Field                | Type   | Required | Description                                                                                                    |
|----------------------|--------|----------|----------------------------------------------------------------------------------------------------------------|
| `dataset_id`         | string | Yes      | Dataset identifier returned by `/upload`                                                                       |
| `input_colorspace`   | string | No       | Colorspace of the captured images. If provided, images are converted from this space to the internal linear space before processing. See [Colorspace Values](#colorspace-values). |
| `output_colorspace`  | string | No       | Desired colorspace for the output images. If provided, images are converted from the internal linear space to this space after reconstruction. See [Colorspace Values](#colorspace-values). |
| `settings`           | object | No       | Processing settings (see [Processing Settings](#processing-settings))                                          |

**Response** `202 Accepted`:
```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "queued",
  "message": "Processing started"
}
```

**Error responses**:

| Code | Reason                                              |
|------|-----------------------------------------------------|
| 400  | Missing `dataset_id`, invalid JSON, or unsupported colorspace value |
| 404  | Dataset not found (upload first)                    |

---

### Job Status

Get the current status and progress of a processing job.

```
GET /api/v1/jobs/{job_id}/status
```

**Authentication**: Required

**Response** `200 OK`:
```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "processing",
  "progress": 65,
  "current_step": "Matching",
  "created_at": "2025-01-15T10:30:00.000000+00:00",
  "updated_at": "2025-01-15T10:35:42.000000+00:00",
  "error": null,
  "settings": { "..." }
}
```

**Status values**:

| Status       | Description                              |
|--------------|------------------------------------------|
| `queued`     | Job is waiting to start                  |
| `processing` | Job is currently running                 |
| `completed`  | Job finished successfully                |
| `failed`     | Job encountered an error                 |
| `cancelled`  | Job was cancelled by user                |

**Progress values** (approximate):

| Progress | Step                        |
|----------|-----------------------------|
| 0-10     | Initializing                |
| 10-30    | ReScan → COLMAP conversion  |
| 30-50    | Feature extraction          |
| 50-65    | Pair generation             |
| 65-80    | Feature matching            |
| 80-100   | SfM / Triangulation         |

---

### Job Logs

Retrieve the processing log for a job.

```
GET /api/v1/jobs/{job_id}/logs
```

**Authentication**: Required

**Response** `200 OK`:
```json
{
  "job_id": "f7e8d9c0b1a2",
  "log": [
    {
      "time": "2025-01-15T10:30:01.000000+00:00",
      "message": "Pipeline started"
    },
    {
      "time": "2025-01-15T10:30:02.000000+00:00",
      "message": "Step 1/5 — ReScan → COLMAP conversion"
    },
    {
      "time": "2025-01-15T10:31:15.000000+00:00",
      "message": "  ✓ 245 images, 1,204,381 LiDAR points"
    }
  ]
}
```

---

### Job Result (Download)

Download the result of a completed job as a ZIP archive.

```
GET /api/v1/jobs/{job_id}/result
```

**Authentication**: Required

**Response** `200 OK`:
- **Content-Type**: `application/zip`
- **Content-Disposition**: `attachment; filename="remap_result_{job_id}.zip"`

The ZIP contains the standard COLMAP output structure:

```
images/
├── 000001.png
├── 000002.png
└── ...
hloc_outputs/
├── features.h5
├── pairs.txt
└── matches.h5
sparse/
└── 0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

**Error responses**:

| Code | Reason                                     |
|------|--------------------------------------------|
| 404  | Job not found                              |
| 409  | Job not completed yet (poll status first)  |

---

### List Jobs

List all jobs (most recent first).

```
GET /api/v1/jobs
```

**Authentication**: Required

**Response** `200 OK`:
```json
{
  "jobs": [
    {
      "job_id": "f7e8d9c0b1a2",
      "status": "completed",
      "progress": 100,
      "current_step": "Done",
      "created_at": "2025-01-15T10:30:00.000000+00:00",
      "updated_at": "2025-01-15T10:45:00.000000+00:00"
    },
    {
      "job_id": "c3d4e5f6a7b8",
      "status": "processing",
      "progress": 45,
      "current_step": "Features",
      "created_at": "2025-01-15T11:00:00.000000+00:00",
      "updated_at": "2025-01-15T11:05:00.000000+00:00"
    }
  ]
}
```

---

### Cancel Job

Cancel a running or queued job.

```
POST /api/v1/jobs/{job_id}/cancel
```

**Authentication**: Required

**Response** `200 OK`:
```json
{
  "job_id": "f7e8d9c0b1a2",
  "status": "cancelled"
}
```

**Error responses**:

| Code | Reason                                           |
|------|--------------------------------------------------|
| 404  | Job not found                                    |
| 409  | Job cannot be cancelled (already completed/failed)|

---

## Processing Settings

All settings are optional. Defaults are used when omitted.

| Setting                  | Type    | Default                  | Description                                                                                                |
|--------------------------|---------|--------------------------|------------------------------------------------------------------------------------------------------------|
| `fps`                    | float   | `5.0`                    | Extraction FPS (frames per second from video)                                                              |
| `feature_type`           | string  | `"superpoint_aachen"`    | Feature detector: `superpoint_aachen`,`superpoint_max`, `disk`, `aliked-n16`,`sift`                        |
| `matcher_type`           | string  | `"superpoint+lightglue"` | Matcher: `superpoint+lightglue`, `superglue`, `disk+lightglue`,`adalam`                                    |
| `max_keypoints`          | int     | `8192`                   | Max keypoints per image                                                                                    |
| `camera_model`           | string  | `"PINHOLE"`              | Camera model: `OPENCV`, `PINHOLE`, `SIMPLE_RADIAL`, `OPENCV_FISHEYE`                                       |
| `mapper_type`            | string  | `"GLOMAP"`               | SfM engine: `COLMAP` or `GLOMAP`                                                                           |
| `stray_approach`         | string  | `"full_sfm"`             | ReScan mode: `full_sfm` or `known_poses` (ARKit)                                                           |
| `pairing_mode`           | string  | `"exhaustive"`           | Pairing: `sequential` or `exhaustive`                                                                      |
| `num_threads`            | int     | *(auto: CPU count)*      | Number of CPU threads                                                                                      |
| `input_colorspace`       | string  |`None`                    | Color Pipeline Input colorspace                                                                            |
| `output_colorspace`      | string  |`None`                    | Color Pipeline Output colorspace                                                                           |
| `stray_confidence`       | int     | `2`                      | LiDAR depth confidence threshold (0-2)                                                                     |
| `stray_depth_subsample`  | int     | `2`                      | Depth frame subsampling factor                                                                             |
| `stray_gen_pointcloud`   | bool    | `true`                   | Generate 3D point cloud from LiDAR depth                                                                   |

### Recommended Settings

| Use Case                 | FPS | Feature            | Matcher               | Approach     |
|--------------------------|-----|--------------------|-----------------------|--------------|
| Indoor walkthrough       | 2-10 |`superpoint_aachen`| `superpoint+lightglue`| `full_sfm`   |
| Outdoor scene            | 3-10 |`superpoint_aachen`| `superpoint+lightglue`| `full_sfm`   |
| Small object / turntable | 2-10 |`superpoint_aachen`| `superpoint+lightglue`| `full_sfm`   |
| Fast-moving scene        | 5-20 |`superpoint_aachen`| `superpoint+lightglue`| `full_sfm`   |

---

## Colorspace Values

The `input_colorspace` and `output_colorspace` top-level fields accept the following string values (case-insensitive):

| Value        | Description                                            |
|--------------|--------------------------------------------------------|
| `linear`     | Linear light sRGB — the pipeline's assumed internal working space |
| `srgb`       | Display sRGB (gamma-corrected, standard monitor output) |
| `acescg`     | ACEScg — ACES CG rendering/compositing working space   |
| `aces2065-1` | ACES2065-1 — ACES interchange / archive format         |
| `rec709`     | Rec. 709 — broadcast/HD television standard            |
| `log`        | Generic logarithmic encoding                           |
| `raw`        | No colorspace interpretation (pass-through)            |

### How colorspace conversion works

- **`input_colorspace`**: If provided, extracted image frames are converted **from** the specified space **to** the internal linear space before feature extraction and SfM reconstruction. If omitted, images are used as-is (assumed to already be in the internal working space).
- **`output_colorspace`**: If provided, the output images are converted **from** the internal linear space **to** the specified space after reconstruction is complete. If omitted, output images remain in the internal working space.
- You can specify `input_colorspace` and `output_colorspace` independently — they do not need to match.

### Example: sRGB input → ACEScg output

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "input_colorspace": "srgb",
  "output_colorspace": "acescg",
  "settings": {
    "fps": 4.0,
    "feature_type": "superpoint_aachen",
    "stray_approach": "full_sfm"
  }
}
```

---

## Error Handling

All errors follow a consistent JSON format:

```json
{
  "error": "Error Type",
  "message": "Detailed description of what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning                                       |
|------|-----------------------------------------------|
| 200  | Success                                       |
| 201  | Resource created (upload)                     |
| 202  | Accepted (processing started)                 |
| 400  | Bad request (invalid input)                   |
| 401  | Unauthorized (missing token)                  |
| 403  | Forbidden (invalid token)                     |
| 404  | Not found                                     |
| 409  | Conflict (job not in expected state)          |
| 413  | Payload too large (max 10 GB)                 |
| 500  | Internal server error                         |

---

## Workflow Example

Complete end-to-end example using `curl`:

```bash
# 1. Check server is running
curl http://localhost:5000/api/v1/health

# 2. Upload a ReScan dataset
curl -X POST http://localhost:5000/api/v1/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@/path/to/rescan_dataset.zip"
# Response: { "dataset_id": "a1b2c3d4e5f6", ... }

# 3. Start processing with custom settings
curl -X POST http://localhost:5000/api/v1/process \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "a1b2c3d4e5f6",
    "settings": {
      "fps": 4.0,
      "feature_type": "superpoint_aachen",
      "stray_approach": "full_sfm"
    }
  }'
# Response: { "job_id": "f7e8d9c0b1a2", "status": "queued", ... }

# 4. Poll job status
curl http://localhost:5000/api/v1/jobs/f7e8d9c0b1a2/status \
  -H "Authorization: Bearer YOUR_API_KEY"
# Response: { "status": "processing", "progress": 65, ... }

# 5. Download result when completed
curl -O -J http://localhost:5000/api/v1/jobs/f7e8d9c0b1a2/result \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## iOS / Swift Integration Guide

Example integration for the ReScan iOS app using `URLSession`.

### Upload Dataset

```swift
func uploadDataset(zipURL: URL, apiKey: String, serverURL: String,
                   completion: @escaping (Result<String, Error>) -> Void) {
    let url = URL(string: "\(serverURL)/api/v1/upload")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)",
                     forHTTPHeaderField: "Content-Type")

    var body = Data()
    let zipData = try! Data(contentsOf: zipURL)
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"dataset.zip\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: application/zip\r\n\r\n".data(using: .utf8)!)
    body.append(zipData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        guard let data = data,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let datasetId = json["dataset_id"] as? String else {
            completion(.failure(NSError(domain: "ReMap", code: -1,
                                       userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
            return
        }
        completion(.success(datasetId))
    }.resume()
}
```

### Start Processing

```swift
func startProcessing(datasetId: String, apiKey: String, serverURL: String,
                     settings: [String: Any] = [:],
                     completion: @escaping (Result<String, Error>) -> Void) {
    let url = URL(string: "\(serverURL)/api/v1/process")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let body: [String: Any] = [
        "dataset_id": datasetId,
        "settings": settings
    ]
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)

    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        guard let data = data,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let jobId = json["job_id"] as? String else {
            completion(.failure(NSError(domain: "ReMap", code: -1,
                                       userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
            return
        }
        completion(.success(jobId))
    }.resume()
}
```

### Poll Job Status

```swift
func pollJobStatus(jobId: String, apiKey: String, serverURL: String,
                   completion: @escaping (String, Int) -> Void) {
    let url = URL(string: "\(serverURL)/api/v1/jobs/\(jobId)/status")!
    var request = URLRequest(url: url)
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

    URLSession.shared.dataTask(with: request) { data, _, _ in
        guard let data = data,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let status = json["status"] as? String,
              let progress = json["progress"] as? Int else { return }
        completion(status, progress)
    }.resume()
}
```

### Download Result

```swift
func downloadResult(jobId: String, apiKey: String, serverURL: String,
                    to destinationURL: URL,
                    completion: @escaping (Result<URL, Error>) -> Void) {
    let url = URL(string: "\(serverURL)/api/v1/jobs/\(jobId)/result")!
    var request = URLRequest(url: url)
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

    URLSession.shared.downloadTask(with: request) { tempURL, response, error in
        if let error = error {
            completion(.failure(error))
            return
        }
        guard let tempURL = tempURL else {
            completion(.failure(NSError(domain: "ReMap", code: -1,
                                       userInfo: [NSLocalizedDescriptionKey: "No data received"])))
            return
        }
        do {
            try FileManager.default.moveItem(at: tempURL, to: destinationURL)
            completion(.success(destinationURL))
        } catch {
            completion(.failure(error))
        }
    }.resume()
}
```

### Complete Workflow

```swift
// Full ReScan → ReMap workflow
let serverURL = "http://192.168.1.100:5000"
let apiKey = "your-api-key-here"

// 1. Upload
uploadDataset(zipURL: capturedDatasetZIP, apiKey: apiKey, serverURL: serverURL) { result in
    switch result {
    case .success(let datasetId):
        // 2. Process
        let settings: [String: Any] = [
            "fps": 4.0,
            "stray_approach": "full_sfm",
            "feature_type": "superpoint_aachen"
        ]
        startProcessing(datasetId: datasetId, apiKey: apiKey,
                        serverURL: serverURL, settings: settings) { result in
            switch result {
            case .success(let jobId):
                // 3. Poll until complete
                Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { timer in
                    pollJobStatus(jobId: jobId, apiKey: apiKey,
                                  serverURL: serverURL) { status, progress in
                        print("Status: \(status), Progress: \(progress)%")
                        if status == "completed" {
                            timer.invalidate()
                            // 4. Download
                            let dest = FileManager.default.temporaryDirectory
                                .appendingPathComponent("result.zip")
                            downloadResult(jobId: jobId, apiKey: apiKey,
                                           serverURL: serverURL, to: dest) { _ in
                                print("Result downloaded!")
                            }
                        } else if status == "failed" {
                            timer.invalidate()
                            print("Processing failed")
                        }
                    }
                }
            case .failure(let error):
                print("Process error: \(error)")
            }
        }
    case .failure(let error):
        print("Upload error: \(error)")
    }
}
```
