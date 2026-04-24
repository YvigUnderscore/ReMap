import { useEffect, useMemo, useState } from 'react'
import { RemapApi } from './api'
import { CameraViewer } from './components/CameraViewer'
import './App.css'
import type { CameraPose, InputMode, JobLogEntry, JobStatus, ProcessingSettings, RuntimeInfo, SettingsSchema } from './types'

type NavTab = 'configuration' | 'processing' | 'server' | 'jobs' | 'advanced'

const FALLBACK_SETTINGS: ProcessingSettings = {
  fps: 4,
  feature_type: 'superpoint_aachen',
  matcher_type: 'superpoint+lightglue',
  max_keypoints: 4096,
  camera_model: 'OPENCV',
  mapper_type: 'COLMAP',
  stray_approach: 'full_sfm',
  pairing_mode: 'sequential',
  num_threads: 8,
  stray_confidence: 2,
  stray_depth_subsample: 2,
  stray_gen_pointcloud: true,
}

const TABS: Array<{ id: NavTab; label: string }> = [
  { id: 'configuration', label: 'Configuration' },
  { id: 'processing', label: 'Traitement' },
  { id: 'server', label: 'Serveur API' },
  { id: 'jobs', label: 'Historique / Jobs' },
  { id: 'advanced', label: 'Paramètres avancés' },
]

function parseMetrics(logs: JobLogEntry[]) {
  const metrics = { images: 0, features: 0, matches: 0, points3d: 0 }

  for (const entry of logs) {
    const msg = entry.message
    const img = msg.match(/([0-9][0-9,]*)\s+images/)
    if (img) metrics.images = Math.max(metrics.images, Number(img[1].replaceAll(',', '')))
    if (msg.includes('Features extracted')) metrics.features = Math.max(metrics.features, 1)
    if (msg.includes('Matching complete')) metrics.matches = Math.max(metrics.matches, 1)

    const pts = msg.match(/([0-9][0-9,]*)\s+(3D points|points LiDAR|LiDAR points)/)
    if (pts) metrics.points3d = Math.max(metrics.points3d, Number(pts[1].replaceAll(',', '')))
  }

  return metrics
}

function toNumber(value: string, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function App() {
  const [tab, setTab] = useState<NavTab>('configuration')
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null)
  const [schema, setSchema] = useState<SettingsSchema | null>(null)
  const [settings, setSettings] = useState<ProcessingSettings>(FALLBACK_SETTINGS)
  const [inputMode, setInputMode] = useState<InputMode>('rescan')
  const [inputColorspace, setInputColorspace] = useState('srgb')
  const [outputColorspace, setOutputColorspace] = useState('acescg')
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [datasetId, setDatasetId] = useState('')
  const [activeJobId, setActiveJobId] = useState('')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [jobLogs, setJobLogs] = useState<JobLogEntry[]>([])
  const [jobs, setJobs] = useState<JobStatus[]>([])
  const [cameras, setCameras] = useState<CameraPose[]>([])
  const [backendLogs, setBackendLogs] = useState<string[]>([])
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  const api = useMemo(() => {
    const baseUrl = runtime?.baseUrl ?? import.meta.env.VITE_REMAP_API_URL ?? 'http://127.0.0.1:5000/api/v1'
    const apiKey = runtime?.apiKey ?? import.meta.env.VITE_REMAP_API_KEY ?? ''
    return new RemapApi(baseUrl, apiKey)
  }, [runtime])

  const metrics = useMemo(() => parseMetrics(jobLogs), [jobLogs])

  useEffect(() => {
    const loadRuntime = async () => {
      try {
        if (window.remapDesktop) {
          const info = await window.remapDesktop.getRuntimeInfo()
          setRuntime(info)
          const logRes = await window.remapDesktop.getBackendLogs()
          setBackendLogs(logRes.logs)
        } else {
          setRuntime({
            platform: navigator.platform,
            isDev: true,
            running: false,
            port: 5000,
            apiKey: import.meta.env.VITE_REMAP_API_KEY ?? '',
            startedAt: '',
            baseUrl: import.meta.env.VITE_REMAP_API_URL ?? 'http://127.0.0.1:5000/api/v1',
          })
        }
      } catch (loadErr) {
        setError(`Runtime init error: ${String(loadErr)}`)
      }
    }

    void loadRuntime()
  }, [])

  useEffect(() => {
    if (!runtime?.apiKey) return
    const loadSchema = async () => {
      try {
        const serverSchema = await api.settingsSchema()
        setSchema(serverSchema)
        setSettings({ ...serverSchema.defaults })
      } catch {
        setSchema(null)
      }
    }
    void loadSchema()
  }, [api, runtime?.apiKey])

  useEffect(() => {
    if (!activeJobId || !runtime?.apiKey) return

    const tick = async () => {
      try {
        const [statusRes, logsRes, jobsRes] = await Promise.all([
          api.status(activeJobId),
          api.logs(activeJobId),
          api.jobs(),
        ])
        setJobStatus(statusRes)
        setJobLogs(logsRes.log)
        setJobs(jobsRes.jobs)

        if (statusRes.status === 'processing' || statusRes.status === 'completed') {
          try {
            const cams = await api.visualizerCameras(activeJobId)
            setCameras(cams.cameras)
          } catch {
            // no live visualization yet
          }
        }
      } catch (pollErr) {
        setError(`Polling error: ${String(pollErr)}`)
      }
    }

    void tick()
    const timer = window.setInterval(() => void tick(), 2500)
    return () => window.clearInterval(timer)
  }, [activeJobId, api, runtime?.apiKey])

  useEffect(() => {
    if (!window.remapDesktop) return

    const timer = window.setInterval(async () => {
      const [statusRes, logsRes] = await Promise.all([
        window.remapDesktop!.getBackendStatus(),
        window.remapDesktop!.getBackendLogs(),
      ])
      setRuntime(statusRes)
      setBackendLogs(logsRes.logs)
    }, 3000)

    return () => window.clearInterval(timer)
  }, [])

  const setField = <K extends keyof ProcessingSettings>(field: K, value: ProcessingSettings[K]) => {
    setSettings((prev) => ({ ...prev, [field]: value }))
  }

  const uploadDataset = async () => {
    if (!zipFile) {
      setError('Sélectionne un ZIP ReScan à uploader.')
      return
    }
    setError('')
    const uploadRes = await api.uploadDataset(zipFile)
    setDatasetId(uploadRes.dataset_id)
    setNotice(`Dataset uploadé: ${uploadRes.dataset_id}`)
  }

  const startProcess = async () => {
    if (!datasetId) {
      setError('Dataset ID manquant. Upload requis.')
      return
    }
    setError('')
    const startRes = await api.startProcessing({
      dataset_id: datasetId,
      settings,
      input_colorspace: inputColorspace,
      output_colorspace: outputColorspace,
    })
    setActiveJobId(startRes.job_id)
    setNotice(`Job démarré: ${startRes.job_id}`)
    setTab('processing')
  }

  const cancelProcess = async () => {
    if (!activeJobId) return
    await api.cancel(activeJobId)
    const statusRes = await api.status(activeJobId)
    setJobStatus(statusRes)
  }

  const refreshJobs = async () => {
    setError('')
    const jobsRes = await api.jobs()
    setJobs(jobsRes.jobs)
  }

  const downloadResult = async (jobId: string) => {
    const blob = await api.downloadResult(jobId)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `remap_result_${jobId}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const checkHealth = async () => {
    setError('')
    const h = await api.health()
    setNotice(`Backend OK (${h.status}) à ${h.timestamp}`)
  }

  const startBackend = async () => {
    if (!window.remapDesktop) return
    const state = await window.remapDesktop.startBackend()
    setRuntime(state)
  }

  const stopBackend = async () => {
    if (!window.remapDesktop) return
    const state = await window.remapDesktop.stopBackend()
    setRuntime(state)
  }

  const options = schema?.options ?? {}
  const features = (options.feature_type as string[] | undefined) ?? ['superpoint_aachen']
  const matchers = (options.matcher_type as string[] | undefined) ?? ['superpoint+lightglue']
  const camerasOpt = (options.camera_model as string[] | undefined) ?? ['OPENCV']
  const mappers = (options.mapper_type as string[] | undefined) ?? ['COLMAP']
  const pairings = (options.pairing_mode as string[] | undefined) ?? ['sequential']
  const approaches = (options.stray_approach as string[] | undefined) ?? ['full_sfm']
  const colorspaces = schema?.supported_colorspaces ?? ['srgb', 'acescg', 'linear']

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>ReMap React Desktop</h1>
          <p>Interface responsive React + backend Python Flask (Windows/Linux)</p>
        </div>
        <div className={`status-dot ${runtime?.running ? 'ok' : 'warn'}`}>
          {runtime?.running ? 'Backend running' : 'Backend offline'}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={tab === item.id ? 'tab active' : 'tab'}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {(notice || error) && (
        <div className="notice-wrap">
          {notice && <div className="notice success">{notice}</div>}
          {error && <div className="notice error">{error}</div>}
        </div>
      )}

      {tab === 'configuration' && (
        <section className="grid two">
          <article className="card">
            <h2>Entrées et mode</h2>
            <label>
              Mode
              <select value={inputMode} onChange={(e) => setInputMode(e.target.value as InputMode)}>
                <option value="video">Video (.mp4, .mov)</option>
                <option value="images">Image Folder</option>
                <option value="rescan">Rescan (LiDAR)</option>
              </select>
            </label>
            <p className="muted">
              En backend actuel API: workflow upload ZIP ReScan. Les modes vidéo/images sont prêts côté UI pour la parité,
              extension backend locale à finaliser.
            </p>
          </article>

          <article className="card">
            <h2>Pipeline SfM</h2>
            <label>
              Feature
              <select value={settings.feature_type} onChange={(e) => setField('feature_type', e.target.value)}>
                {features.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              Matcher
              <select value={settings.matcher_type} onChange={(e) => setField('matcher_type', e.target.value)}>
                {matchers.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              Camera model
              <select value={settings.camera_model} onChange={(e) => setField('camera_model', e.target.value)}>
                {camerasOpt.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              Mapper
              <select value={settings.mapper_type} onChange={(e) => setField('mapper_type', e.target.value)}>
                {mappers.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              Pairing
              <select value={settings.pairing_mode} onChange={(e) => setField('pairing_mode', e.target.value)}>
                {pairings.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
          </article>

          <article className="card">
            <h2>Extraction et performance</h2>
            <label>
              FPS
              <input type="number" min={0.5} step={0.1} value={settings.fps} onChange={(e) => setField('fps', toNumber(e.target.value, settings.fps))} />
            </label>
            <label>
              Max keypoints
              <input type="number" min={256} step={256} value={settings.max_keypoints} onChange={(e) => setField('max_keypoints', toNumber(e.target.value, settings.max_keypoints))} />
            </label>
            <label>
              Threads
              <input type="number" min={1} step={1} value={settings.num_threads} onChange={(e) => setField('num_threads', toNumber(e.target.value, settings.num_threads))} />
            </label>
          </article>

          <article className="card">
            <h2>LiDAR / ReScan + color management</h2>
            <label>
              Approach
              <select value={settings.stray_approach} onChange={(e) => setField('stray_approach', e.target.value)}>
                {approaches.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              LiDAR confidence
              <input type="number" min={0} max={2} step={1} value={settings.stray_confidence} onChange={(e) => setField('stray_confidence', toNumber(e.target.value, settings.stray_confidence))} />
            </label>
            <label>
              Depth subsample
              <input type="number" min={1} step={1} value={settings.stray_depth_subsample} onChange={(e) => setField('stray_depth_subsample', toNumber(e.target.value, settings.stray_depth_subsample))} />
            </label>
            <label className="toggle">
              <input type="checkbox" checked={settings.stray_gen_pointcloud} onChange={(e) => setField('stray_gen_pointcloud', e.target.checked)} />
              Generate LiDAR pointcloud
            </label>
            <label>
              Input colorspace
              <select value={inputColorspace} onChange={(e) => setInputColorspace(e.target.value)}>
                {colorspaces.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              Output colorspace
              <select value={outputColorspace} onChange={(e) => setOutputColorspace(e.target.value)}>
                {colorspaces.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
          </article>
        </section>
      )}

      {tab === 'processing' && (
        <section className="grid two">
          <article className="card">
            <h2>Démarrage job</h2>
            <label>
              ZIP dataset ReScan
              <input type="file" accept=".zip" onChange={(e) => setZipFile(e.target.files?.[0] ?? null)} />
            </label>
            <div className="row">
              <button onClick={() => void uploadDataset()}>Upload dataset</button>
              <input value={datasetId} onChange={(e) => setDatasetId(e.target.value)} placeholder="dataset_id" />
              <button onClick={() => void startProcess()}>Start processing</button>
            </div>
            <div className="row">
              <input value={activeJobId} onChange={(e) => setActiveJobId(e.target.value)} placeholder="job_id à suivre" />
              <button className="danger" onClick={() => void cancelProcess()}>Cancel</button>
            </div>
            <div className="progress-wrap">
              <div className="progress-label">{jobStatus?.current_step ?? 'Waiting...'}</div>
              <div className="progress">
                <div className="progress-inner" style={{ width: `${jobStatus?.progress ?? 0}%` }} />
              </div>
              <div className="progress-meta">Status: {jobStatus?.status ?? 'n/a'} · {jobStatus?.progress ?? 0}%</div>
            </div>
          </article>

          <article className="card">
            <h2>Visualizers opérationnels</h2>
            <div className="metrics">
              <div><strong>{metrics.images}</strong><span>Images</span></div>
              <div><strong>{metrics.features}</strong><span>Features step</span></div>
              <div><strong>{metrics.matches}</strong><span>Matching step</span></div>
              <div><strong>{metrics.points3d.toLocaleString()}</strong><span>Points 3D</span></div>
            </div>
            <h3>Caméra cloud (live)</h3>
            <CameraViewer cameras={cameras} />
          </article>

          <article className="card full">
            <h2>Console job (filtrable par backend)</h2>
            <pre>{jobLogs.map((l) => `[${l.time}] ${l.message}`).join('\n') || 'No logs yet.'}</pre>
          </article>
        </section>
      )}

      {tab === 'server' && (
        <section className="grid two">
          <article className="card">
            <h2>Serveur API embarqué</h2>
            <p><strong>URL:</strong> {runtime?.baseUrl}</p>
            <p><strong>API Key:</strong> <code>{runtime?.apiKey || '(none)'}</code></p>
            <p><strong>Platform:</strong> {runtime?.platform}</p>
            <div className="row">
              <button onClick={() => void checkHealth()}>Health check</button>
              <button onClick={() => void startBackend()} disabled={!window.remapDesktop}>Start backend</button>
              <button className="danger" onClick={() => void stopBackend()} disabled={!window.remapDesktop}>Stop backend</button>
            </div>
            {!window.remapDesktop && (
              <p className="muted">Mode web: contrôle process backend non disponible (utiliser Electron).</p>
            )}
          </article>
          <article className="card">
            <h2>Logs backend Python</h2>
            <pre>{backendLogs.join('\n') || 'No backend logs.'}</pre>
          </article>
        </section>
      )}

      {tab === 'jobs' && (
        <section className="grid two">
          <article className="card full">
            <div className="row between">
              <h2>Historique jobs</h2>
              <button onClick={() => void refreshJobs()}>Refresh</button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>{job.job_id}</td>
                    <td>{job.status}</td>
                    <td>{job.progress}%</td>
                    <td>{job.updated_at}</td>
                    <td className="actions">
                      <button onClick={() => setActiveJobId(job.job_id)}>Track</button>
                      <button onClick={() => void downloadResult(job.job_id)} disabled={job.status !== 'completed'}>Download</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        </section>
      )}

      {tab === 'advanced' && (
        <section className="grid two">
          <article className="card">
            <h2>Parité paramètres (schema backend)</h2>
            <pre>{JSON.stringify(schema, null, 2) || 'Schema not available yet.'}</pre>
          </article>
          <article className="card">
            <h2>État migration</h2>
            <ul>
              <li>✅ Shell desktop Electron + navigateur embarqué fluide</li>
              <li>✅ Backend Python local démarré/arrêté depuis l’app</li>
              <li>✅ UI responsive multi-sections (config, process, server, jobs, advanced)</li>
              <li>✅ Monitoring live: progression, logs, métriques, visualizer caméras</li>
              <li>⚠️ Workflow API natif actuel centré ReScan ZIP; extension vidéo/images locale à finaliser côté backend.</li>
            </ul>
          </article>
        </section>
      )}
    </div>
  )
}

export default App
