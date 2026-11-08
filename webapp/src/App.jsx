import { useEffect, useMemo, useState } from "react";
import SliceGrid from "./components/SliceGrid.jsx";

const DEFAULTS = {
  data_source: "synthetic",
  walnut_dose: "10",
  hku_case: "HN-CHUM-006",
  hku_scanner: "iba",
  volume_size: 48,
  n_angles: 60,
  detector_size: 64,
  noise_sigma: 0.025,
  denoise_mode: "auto",
  device: "CPU",
};

function sourceSelectValue(form) {
  if (form.data_source === "walnut") return `walnut_${form.walnut_dose}`;
  if (form.data_source === "hku_hn") {
    return `hku_${form.hku_scanner}_${form.hku_case}`;
  }
  return "synthetic";
}

export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [live, setLive] = useState({
    status: "idle",
    progress: 0,
    message: "Ready",
    error: null,
    has_result: false,
  });
  const [result, setResult] = useState(null);
  const [telemetry, setTelemetry] = useState({ cpu_percent: 0, memory_percent: 0 });
  const [devices, setDevices] = useState(["CPU"]);
  const [datasets, setDatasets] = useState(null);
  const [models, setModels] = useState({ stub: false, redcnn: false });

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        if (d.devices?.length) setDevices(d.devices);
        if (d.models) setModels(d.models);
      })
      .catch(() => {});

    fetch("/api/datasets")
      .then((r) => r.json())
      .then(setDatasets)
      .catch(() => {});

    const es = new EventSource("/api/events");
    es.onmessage = (ev) => {
      try {
        setLive(JSON.parse(ev.data));
      } catch {
        /* ignore */
      }
    };
    const tel = setInterval(() => {
      fetch("/api/telemetry")
        .then((r) => r.json())
        .then(setTelemetry)
        .catch(() => {});
    }, 2000);
    return () => {
      es.close();
      clearInterval(tel);
    };
  }, []);

  useEffect(() => {
    if (live.status === "done" && live.has_result) {
      fetch("/api/result")
        .then((r) => r.json())
        .then(setResult)
        .catch(() => {});
    }
  }, [live.status, live.has_result]);

  const busy = live.status === "running";
  const isWalnut = form.data_source === "walnut";
  const isHku = form.data_source === "hku_hn";
  const isReal = isWalnut || isHku;
  const sources = datasets?.sources ?? [
    { id: "synthetic", label: "Synthetic phantom", available: true, data_source: "synthetic" },
  ];
  const selectedId = sourceSelectValue(form);
  const selected =
    sources.find((s) => s.id === selectedId) ||
    sources.find((s) => s.data_source === form.data_source && s.available) ||
    sources[0];
  const selectedReady = selected ? selected.available !== false : !isReal;

  const kpis = useMemo(() => {
    if (!result) return [];
    const psnr =
      result.quality?.psnr_denoise_vs_phantom ??
      result.quality?.snr_denoise ??
      "—";
    const psnrLabel = result.quality?.psnr_denoise_vs_phantom != null ? "PSNR↑" : "SNR";
    let sourceLabel = "Synthetic";
    if (result.source === "walnut") {
      sourceLabel = `Walnut ${result.walnut?.dose ?? ""}%`.trim();
    } else if (result.source === "hku_hn") {
      sourceLabel = `HKU ${result.hku?.case_id ?? "HN"}`;
    }
    return [
      { label: "Source", value: sourceLabel },
      { label: "Total", value: `${result.timings_total_ms} ms` },
      { label: "FDK", value: `${result.timings_ms?.fdk_ms ?? "—"} ms` },
      { label: "Denoise", value: `${result.timings_ms?.denoise_ms ?? "—"} ms` },
      { label: "Backend", value: result.denoise?.backend ?? "—" },
      { label: "Device", value: result.denoise?.device ?? form.device },
      { label: psnrLabel, value: typeof psnr === "number" ? `${psnr}` : String(psnr) },
      { label: "CPU", value: `${telemetry.cpu_percent?.toFixed?.(0) ?? 0}%` },
    ];
  }, [result, telemetry, form.device]);

  function update(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function onSourceChange(id) {
    const src = sources.find((s) => s.id === id);
    if (!src) return;
    if (src.data_source === "walnut") {
      setForm((f) => ({
        ...f,
        data_source: "walnut",
        walnut_dose: src.walnut_dose || id.replace("walnut_", ""),
      }));
    } else if (src.data_source === "hku_hn") {
      setForm((f) => ({
        ...f,
        data_source: "hku_hn",
        hku_case: src.hku_case || "HN-CHUM-006",
        hku_scanner: src.hku_scanner || "iba",
      }));
    } else {
      setForm((f) => ({ ...f, data_source: "synthetic" }));
    }
  }

  async function onRun() {
    setResult(null);
    await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
  }

  const attribution = result?.walnut?.attribution || result?.hku?.attribution;
  const stageHint =
    isHku && !selectedReady
      ? datasets?.stage_hku_hint || "python scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006"
      : null;

  return (
    <div className="app">
      <header className="hero">
        <h1 className="brand">CBCT FDK Denoise</h1>
        <p className="tagline">
          Reference cone-beam CT pipeline: synthetic phantom, Zenodo walnut doses, or HKU
          head-and-neck projections (MIT), with FDK and RED-CNN / stub / classical denoise.
        </p>
        <p className="disclaimer">
          Not for clinical or diagnostic use — evaluation sample only.
        </p>
      </header>

      <div className="layout">
        <aside className="panel">
          <h2>Configuration</h2>
          <div className="field">
            <label>Dataset</label>
            <select value={selected?.id || "synthetic"} onChange={(e) => onSourceChange(e.target.value)}>
              {sources.map((s) => (
                <option key={s.id} value={s.id} disabled={s.available === false}>
                  {s.label}
                  {s.available === false ? " (not staged)" : ""}
                </option>
              ))}
            </select>
          </div>

          {isWalnut && (
            <p className="status" style={{ marginTop: 0, marginBottom: 12 }}>
              Zenodo walnut · CC BY 4.0
            </p>
          )}
          {isHku && (
            <p className="status" style={{ marginTop: 0, marginBottom: 12 }}>
              {selectedReady
                ? `HKU MedIP CBCT · MIT · ${form.hku_scanner}/hn/${form.hku_case}`
                : `Not staged. ${stageHint}`}
            </p>
          )}

          <div className="field">
            <label>Volume size</label>
            <input
              type="number"
              min={32}
              max={96}
              value={form.volume_size}
              onChange={(e) => update("volume_size", Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>{isReal ? "Max angles (subsample)" : "Projection angles"}</label>
            <input
              type="number"
              min={30}
              max={360}
              value={form.n_angles}
              onChange={(e) => update("n_angles", Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>{isReal ? "Detector downsample" : "Detector size"}</label>
            <input
              type="number"
              min={48}
              max={256}
              value={form.detector_size}
              onChange={(e) => update("detector_size", Number(e.target.value))}
            />
          </div>
          {!isReal && (
            <div className="field">
              <label>Noise σ</label>
              <input
                type="number"
                step="0.005"
                min={0}
                max={0.1}
                value={form.noise_sigma}
                onChange={(e) => update("noise_sigma", Number(e.target.value))}
              />
            </div>
          )}
          <div className="field">
            <label>Denoise mode</label>
            <select
              value={form.denoise_mode}
              onChange={(e) => update("denoise_mode", e.target.value)}
            >
              <option value="auto">auto (RED-CNN → stub → classical)</option>
              <option value="redcnn" disabled={!models.redcnn}>
                redcnn{models.redcnn ? "" : " (not staged)"}
              </option>
              <option value="openvino" disabled={!models.stub}>
                openvino stub{models.stub ? "" : " (not staged)"}
              </option>
              <option value="classical">classical</option>
            </select>
          </div>
          <div className="field">
            <label>Device</label>
            <select value={form.device} onChange={(e) => update("device", e.target.value)}>
              {devices.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
              {!devices.includes("AUTO") && <option value="AUTO">AUTO</option>}
            </select>
          </div>
          <div className="actions">
            <button className="primary" disabled={busy || !selectedReady} onClick={onRun}>
              {busy ? "Running…" : "Start reconstruction"}
            </button>
          </div>
          <div className="status">
            {live.message}
            {live.error ? ` — ${live.error}` : ""}
            <div className="bar">
              <span style={{ width: `${live.progress || 0}%` }} />
            </div>
          </div>
        </aside>

        <main className="panel">
          <h2>Results</h2>
          {result ? (
            <>
              <div className="kpi-grid">
                {kpis.map((k) => (
                  <div className="kpi" key={k.label}>
                    <div className="label">{k.label}</div>
                    <div className="value">{k.value}</div>
                  </div>
                ))}
              </div>
              {attribution && (
                <p className="status" style={{ marginBottom: 14 }}>
                  Data: {attribution}
                </p>
              )}
              <SliceGrid result={result} />
            </>
          ) : (
            <div className="empty">
              Run a reconstruction to view axial / coronal / sagittal slices and KPIs.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
