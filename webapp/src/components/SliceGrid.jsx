function Frame({ title, subtitle, b64 }) {
  return (
    <div className="view">
      <header>
        <span>{title}</span>
        <span>{subtitle}</span>
      </header>
      {b64 ? (
        <img src={`data:image/png;base64,${b64}`} alt={`${title} ${subtitle}`} />
      ) : (
        <div className="empty">No image</div>
      )}
    </div>
  );
}

export default function SliceGrid({ result }) {
  const stages =
    result.source === "synthetic"
      ? [
          { key: "phantom", label: "Phantom" },
          { key: "recon", label: "FDK recon" },
          { key: "denoised", label: "Denoised" },
        ]
      : [
          { key: "recon", label: "FDK recon" },
          { key: "denoised", label: "Denoised" },
        ];
  const views = ["axial", "coronal", "sagittal"];
  let projSub = "mid-angle";
  if (result.source === "walnut" && result.walnut?.dose) {
    projSub = `walnut ${result.walnut.dose}%`;
  } else if (result.source === "hku_hn" && result.hku?.case_id) {
    projSub = `HKU ${result.hku.case_id}`;
  }

  return (
    <div className="views">
      {result.projection_preview && (
        <div className="view-row" style={{ gridTemplateColumns: "1fr" }}>
          <Frame title="Projection" subtitle={projSub} b64={result.projection_preview} />
        </div>
      )}
      {stages.map((stage) => (
        <div key={stage.key}>
          <div className="view-row">
            {views.map((view) => (
              <Frame
                key={view}
                title={stage.label}
                subtitle={view}
                b64={result.slices?.[stage.key]?.[view]}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
