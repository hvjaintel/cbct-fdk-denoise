import { useCallback, useEffect, useRef, useState } from "react";

const AXES = ["axial", "coronal", "sagittal"];

export default function VolumeViewer({ resultKey }) {
  const [meta, setMeta] = useState(null);
  const [stage, setStage] = useState("denoised");
  const [mode, setMode] = useState("scrub"); // scrub | mip
  const [axis, setAxis] = useState("axial");
  const [index, setIndex] = useState(0);
  const [angle, setAngle] = useState(0);
  const [png, setPng] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  const playRef = useRef(null);

  const stageInfo = meta?.stages?.[stage] || meta?.stages?.recon || meta?.stages?.denoised;
  const maxIndex = stageInfo ? Math.max((stageInfo.axes?.[axis] || 1) - 1, 0) : 0;

  const loadMeta = useCallback(() => {
    fetch("/api/volume")
      .then((r) => {
        if (!r.ok) throw new Error("No volume yet");
        return r.json();
      })
      .then((data) => {
        setMeta(data);
        setError(null);
        const stages = Object.keys(data.stages || {});
        const preferred = stages.includes("denoised")
          ? "denoised"
          : stages.includes("recon")
            ? "recon"
            : stages[0];
        if (preferred) setStage(preferred);
        if (preferred && data.stages[preferred]) {
          setIndex(Math.floor(data.stages[preferred].axes.axial / 2));
        }
      })
      .catch((e) => {
        setMeta(null);
        setError(e.message || "No volume");
      });
  }, []);

  useEffect(() => {
    loadMeta();
  }, [loadMeta, resultKey]);

  useEffect(() => {
    if (!meta?.has_volume) return undefined;
    let cancelled = false;
    const url =
      mode === "mip"
        ? `/api/volume/mip?stage=${encodeURIComponent(stage)}&angle=${angle}`
        : `/api/volume/slice?stage=${encodeURIComponent(stage)}&axis=${encodeURIComponent(axis)}&index=${index}`;

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error("Slice request failed");
        return r.json();
      })
      .then((data) => {
        if (!cancelled) {
          setPng(data.png);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || "Failed to load slice");
      });
    return () => {
      cancelled = true;
    };
  }, [meta, stage, mode, axis, index, angle]);

  useEffect(() => {
    if (!playing || !meta?.has_volume) {
      if (playRef.current) {
        clearInterval(playRef.current);
        playRef.current = null;
      }
      return undefined;
    }
    playRef.current = setInterval(() => {
      if (mode === "mip") {
        setAngle((a) => (a + 6) % 360);
      } else {
        setIndex((i) => (i >= maxIndex ? 0 : i + 1));
      }
    }, 120);
    return () => {
      if (playRef.current) clearInterval(playRef.current);
    };
  }, [playing, mode, maxIndex, meta]);

  if (error && !meta) {
    return (
      <div className="volume-viewer">
        <h3>3D viewer</h3>
        <div className="empty">Run a reconstruction to scrub the volume.</div>
      </div>
    );
  }

  const stages = Object.keys(meta?.stages || {});

  return (
    <div className="volume-viewer">
      <div className="volume-header">
        <h3>3D viewer</h3>
        <div className="volume-toggles">
          <button
            type="button"
            className={mode === "scrub" ? "chip active" : "chip"}
            onClick={() => {
              setMode("scrub");
              setPlaying(false);
            }}
          >
            Slice scrubber
          </button>
          <button
            type="button"
            className={mode === "mip" ? "chip active" : "chip"}
            onClick={() => {
              setMode("mip");
              setPlaying(false);
            }}
          >
            Spinning MIP
          </button>
        </div>
      </div>

      <div className="volume-controls">
        <div className="field inline">
          <label>Stage</label>
          <select value={stage} onChange={(e) => setStage(e.target.value)}>
            {stages.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {mode === "scrub" ? (
          <>
            <div className="axis-row">
              {AXES.map((a) => (
                <button
                  key={a}
                  type="button"
                  className={axis === a ? "chip active" : "chip"}
                  onClick={() => {
                    setAxis(a);
                    const mid = Math.floor((stageInfo?.axes?.[a] || 1) / 2);
                    setIndex(mid);
                  }}
                >
                  {a}
                </button>
              ))}
            </div>
            <div className="field">
              <label>
                Slice {index + 1} / {maxIndex + 1}
              </label>
              <input
                type="range"
                min={0}
                max={maxIndex}
                value={Math.min(index, maxIndex)}
                onChange={(e) => setIndex(Number(e.target.value))}
              />
            </div>
          </>
        ) : (
          <div className="field">
            <label>Angle {Math.round(angle)}°</label>
            <input
              type="range"
              min={0}
              max={359}
              value={Math.round(angle)}
              onChange={(e) => setAngle(Number(e.target.value))}
            />
          </div>
        )}

        <div className="actions">
          <button type="button" className="ghost" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
        </div>
      </div>

      <div className="volume-frame">
        {png ? (
          <img src={`data:image/png;base64,${png}`} alt={`${stage} ${mode}`} />
        ) : (
          <div className="empty">Loading…</div>
        )}
        {error && <div className="status">{error}</div>}
      </div>
    </div>
  );
}
