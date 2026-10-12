import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

// SONARIS — AI-Powered Underwater Anomaly Intelligence
// Replace the contents of frontend/src/App.jsx with this file.

function App() {
  const [page, setPage] = useState("dashboard");
  const [sonarFile, setSonarFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [heading, setHeading] = useState("");
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [memoryRecords, setMemoryRecords] = useState([]);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState("");

  const detections = analysisResult?.detections || [];

  const getRiskLevel = (risk = 0) => {
    const value = Number(risk) || 0;
    if (value >= 70) return "HIGH";
    if (value >= 40) return "MEDIUM";
    return "LOW";
  };

  const getRiskClass = (risk = 0) => {
    const value = Number(risk) || 0;
    if (value >= 70) return "high";
    if (value >= 40) return "medium";
    return "low";
  };

  const formatLabel = (label) => {
    if (!label) return "UNKNOWN ANOMALY";

    return String(label)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  };

  const formatConfidence = (confidence = 0) => {
    let value = Number(confidence) || 0;

    // Supports either 0.91 or 91 from the backend.
    if (value > 1) value = value / 100;

    return `${(value * 100).toFixed(1)}%`;
  };

  const getRisk = (target) => {
    const value =
      target?.risk_rate_percent ??
      target?.risk_score ??
      target?.risk ??
      0;

    return Number(value) || 0;
  };

  const getVerification = (target) => {
    const verification = target?.multi_step_verification || {};

    return {
      gate1:
        verification.gate1_confidence_pass ??
        verification.gate1_confidence ??
        false,
      gate2:
        verification.gate2_shadow_confirmed ??
        verification.gate2_shadow_confirmation ??
        false,
      status:
        verification.gate3_status ??
        verification.status ??
        "UNKNOWN",
    };
  };

  const getCoordinates = (target) => {
    const coords =
      target?.seabed_memory_coords ||
      target?.seafloor_memory_coords ||
      {};

    return {
      lat: coords.lat ?? coords.latitude ?? "—",
      lon: coords.lon ?? coords.longitude ?? "—",
    };
  };

  const sameTarget = (a, b) => {
    if (!a || !b) return false;

    if (a.bbox && b.bbox) {
      return JSON.stringify(a.bbox) === JSON.stringify(b.bbox);
    }

    return a === b;
  };

  const normalizeResult = (data) => {
    if (!data || typeof data !== "object") {
      return {
        status: "success",
        filename: sonarFile?.name || "sonar-image",
        telemetry: {
          latitude: latitude || "12.2958",
          longitude: longitude || "76.6394",
          heading: heading || "0",
        },
        total_targets: 0,
        detections: [],
      };
    }

    // Current SONARIS JSON shape.
    const normalized = {
      ...data,
      detections: Array.isArray(data.detections) ? data.detections : [],
      total_targets:
        data.total_targets ??
        (Array.isArray(data.detections) ? data.detections.length : 0),
    };

    // Older backend used telemetry_received.
    if (!normalized.telemetry && normalized.telemetry_received) {
      normalized.telemetry = {
        latitude: normalized.telemetry_received.lat,
        longitude: normalized.telemetry_received.lon,
        heading: normalized.telemetry_received.heading,
      };
    }

    // Older/alternate field names.
    if (!normalized.telemetry) {
      normalized.telemetry = {
        latitude: latitude || "12.2958",
        longitude: longitude || "76.6394",
        heading: heading || "0",
      };
    }

    return normalized;
  };

  const loadMemory = async () => {
  try {
    const response = await fetch(`${API_BASE}/memory`);
    const data = await response.json();

    setMemoryRecords(Array.isArray(data) ? data : []);
  } catch (error) {
    console.error("Failed to load seafloor memory:", error);
  }
};

  const analyzeSonar = async () => {
    if (!sonarFile) {
      setError("Please select a sonar image first.");
      return;
    }

    setAnalyzing(true);
    setError("");
    setSelectedTarget(null);

    const formData = new FormData();
    formData.append("file", sonarFile);

    // Send both naming conventions used by the SONARIS backend versions.
    formData.append("latitude", latitude || "12.2958");
    formData.append("longitude", longitude || "76.6394");
    formData.append("heading", heading || "0");
    formData.append("lat", latitude || "12.2958");
    formData.append("lon", longitude || "76.6394");

    try {
      /*
       * The JSON analysis endpoint is preferred because the dashboard needs
       * detections, confidence, risk, coordinates and verification data.
       */
      let response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      /*
       * Some SONARIS backend versions expose /detect/visualize instead.
       * If /analyze is unavailable, try that endpoint automatically.
       */
      if (response.status === 404) {
        response = await fetch(`${API_BASE}/detect/visualize`, {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        let detail = "";

        try {
          const errorData = await response.json();
          detail = errorData?.detail
            ? ` — ${errorData.detail}`
            : "";
        } catch {
          // Ignore non-JSON error bodies.
        }

        throw new Error(
          `SONARIS backend returned HTTP ${response.status}${detail}`
        );
      }

      const contentType = response.headers.get("content-type") || "";

      /*
       * JSON response: full AI result.
       */
      if (contentType.includes("application/json")) {
        const data = await response.json();
        const normalized = normalizeResult(data);

        console.log("SONARIS AI RESULT:", normalized);
        console.log(
          "DETECTIONS:",
          JSON.stringify(normalized.detections, null, 2)
        );

        setAnalysisResult(normalized);
        loadMemory();
        setPage("results");
        return;
      }

      /*
       * Image response: visualization-only backend.
       * We still display the returned sonar visualization instead of
       * falsely claiming that the backend is offline.
       */
      if (contentType.startsWith("image/")) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        const visualizationResult = {
          status: "success",
          filename: sonarFile.name,
          telemetry: {
            latitude: latitude || "12.2958",
            longitude: longitude || "76.6394",
            heading: heading || "0",
          },
          total_targets: 0,
          detections: [],
          visual_render_url: imageUrl,
          visualization_only: true,
        };

        setAnalysisResult(visualizationResult);
        setPage("results");
        return;
      }

      throw new Error("Unexpected response received from SONARIS backend.");
    } catch (err) {
      console.error("SONARIS ANALYSIS ERROR:", err);

      if (err instanceof TypeError) {
        setError(
          "Cannot reach the SONARIS backend. Start FastAPI on port 8000 and try again."
        );
      } else {
        setError(err.message || "SONARIS analysis failed.");
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const startNewSurvey = () => {
    setPage("new-survey");
    setError("");
    setSelectedTarget(null);
  };

  const openTargets = () => {
    if (analysisResult) {
      setPage("results");
    } else {
      setPage("new-survey");
    }
  };

  const Sidebar = () => (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">S</div>

        <div>
          <h2>SONARIS</h2>
          <span>UNDERWATER INTELLIGENCE</span>
        </div>
      </div>

      <nav className="nav">
        <button
          className={page === "dashboard" ? "active" : ""}
          onClick={() => setPage("dashboard")}
        >
          <span>⌂</span>
          Dashboard
        </button>

        <button
          className={page === "new-survey" ? "active" : ""}
          onClick={startNewSurvey}
        >
          <span>＋</span>
          New Survey
        </button>

        <button
          className={page === "results" ? "active" : ""}
          onClick={openTargets}
        >
          <span>◈</span>
          Targets
        </button>

        <button onClick={() => setPage("memory")}>
          <span>◉</span>
          Seafloor Memory
        </button>

        <button
  className={page === "change-detection" ? "active" : ""}
  onClick={() => setPage("change-detection")}
>
  <span>↔</span>
  Change Detection
</button>

        <button onClick={() => setPage("dashboard")}>
          <span>⚠</span>
          Risk Prioritization
        </button>
      </nav>

      <div className="system-status">
        <span className="status-dot" />

        <div>
          <strong>System Online</strong>
          <small>AI Engine Ready</small>
        </div>
      </div>
    </aside>
  );

  const Topbar = ({ title, subtitle, status = "API Connected" }) => (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>

      <div className="api-status">
        <span className="status-dot" />
        {status}
      </div>
    </header>
  );

  const TargetDetails = ({ target, index }) => {
    if (!target) return null;

    const risk = getRisk(target);
    const verification = getVerification(target);
    const coords = getCoordinates(target);

    return (
      <div className="target-details-panel">
        <div className="target-details-header">
          <div>
            <span className="eyebrow">TARGET INTELLIGENCE</span>

            <h2>
              T{String(index + 1).padStart(3, "0")} —{" "}
              {formatLabel(target.label)}
            </h2>
          </div>

          <button
            className="close-details"
            onClick={() => setSelectedTarget(null)}
            aria-label="Close target details"
          >
            ×
          </button>
        </div>

        <div className="target-detail-grid">
          <div className="detail-box">
            <span>AI CONFIDENCE</span>
            <strong>
              {formatConfidence(target.confidence_rate)}
            </strong>
          </div>

          <div className={`detail-box detail-risk ${getRiskClass(risk)}`}>
            <span>RISK SCORE</span>
            <strong>{risk.toFixed(1)}%</strong>
            <small>{getRiskLevel(risk)}</small>
          </div>

          <div className="detail-box">
            <span>VERIFICATION</span>
            <strong>{verification.status}</strong>
          </div>
          <div className="detail-box">
  <span>SEAFLOOR MEMORY MATCH</span>
  <strong>{target.comparison_status || "N/A"}</strong>
  {target.previous_target_id && (
    <small>
      Previous Target: {target.previous_target_id}
    </small>
  )}
</div>

          <div className="detail-box">
            <span>TARGET TYPE</span>
            <strong>{formatLabel(target.label)}</strong>
          </div>
        </div>

        <div className="verification-section">
          <div className="section-title">
            <span className="section-icon">✓</span>

            <div>
              <h2>Multi-Step Verification</h2>
              <p>AI evidence validation pipeline</p>
            </div>
          </div>

          <div className="verification-row">
            <div className="gate-number">01</div>

            <div>
              <strong>Confidence Pass</strong>
              <span>
                Model confidence threshold check
              </span>
            </div>

            <b
              className={
                verification.gate1
                  ? "verification-pass"
                  : "verification-fail"
              }
            >
              {verification.gate1 ? "PASS" : "FAIL"}
            </b>
          </div>

          <div className="verification-row">
            <div className="gate-number">02</div>

            <div>
              <strong>Shadow Confirmation</strong>
              <span>
                Acoustic shadow evidence check
              </span>
            </div>

            <b
              className={
                verification.gate2
                  ? "verification-pass"
                  : "verification-fail"
              }
            >
              {verification.gate2 ? "PASS" : "FAIL"}
            </b>
          </div>

          <div className="verification-row">
            <div className="gate-number">03</div>

            <div>
              <strong>Final Classification</strong>
              <span>
                Multi-evidence target decision
              </span>
            </div>

            <b
              className={
                verification.status === "VERIFIED"
                  ? "verification-pass"
                  : "verification-fail"
              }
            >
              {verification.status}
            </b>
          </div>
        </div>

        <div className="coordinates-section">
          <div className="section-title">
            <span className="section-icon">⌖</span>

            <div>
              <h2>Seafloor Memory</h2>
              <p>Target geographic reference</p>
            </div>
          </div>

          <div className="coordinate-grid">
            <div className="coordinate-box">
              <span>LATITUDE</span>
              <strong>{coords.lat}</strong>
            </div>

            <div className="coordinate-box">
              <span>LONGITUDE</span>
              <strong>{coords.lon}</strong>
            </div>
          </div>
        </div>

        {target.bbox && (
          <div className="bbox-section">
            <span>BOUNDING BOX</span>

            <div className="bbox-content">
              [{target.bbox.join(", ")}]
            </div>
          </div>
        )}

        <div className="target-action-bar">
          <div>
            <span>RISK ASSESSMENT</span>

            <strong className={`risk-text ${getRiskClass(risk)}`}>
              {getRiskLevel(risk)} PRIORITY
            </strong>
          </div>

          <div>
            <span>OPERATOR ACTION</span>

            <strong>
              {getRiskLevel(risk) === "HIGH"
                ? "Inspect / Escalate"
                : getRiskLevel(risk) === "MEDIUM"
                ? "Review Target"
                : "Monitor"}
            </strong>
          </div>
        </div>
      </div>
    );
  };

  const TargetList = ({ items }) => (
    <div className="target-list">
      {items.length === 0 ? (
        <div
          style={{
            padding: "30px 10px",
            textAlign: "center",
            opacity: 0.7,
          }}
        >
          No targets detected.
        </div>
      ) : (
        items.map((target, index) => {
          const risk = getRisk(target);
          const riskClass = getRiskClass(risk);
          const verification = getVerification(target);

          return (
            <div
              className={`target-item ${
                sameTarget(selectedTarget, target)
                  ? "selected"
                  : ""
              }`}
              key={`${target.label || "target"}-${index}`}
              onClick={() => setSelectedTarget(target)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  setSelectedTarget(target);
                }
              }}
            >
              <div className="target-id">
                T{String(index + 1).padStart(3, "0")}
              </div>

              <div className="target-info">
                <strong>{formatLabel(target.label)}</strong>

                <span>
                  {formatConfidence(target.confidence_rate)}
                  {" "}confidence
                </span>

                <span
                  style={{
                    marginTop: "4px",
                    opacity: 0.65,
                  }}
                >
                  Verification: {verification.status}
                </span>
              </div>

              <span className={`risk ${riskClass}`}>
                {getRiskLevel(risk)}
              </span>
            </div>
          );
        })
      )}
    </div>
  );

  // ------------------------------------------------------------
  // NEW SURVEY
  // ------------------------------------------------------------
  if (page === "new-survey") {
    return (
      <div className="app">
        <Sidebar />

        <main className="main-content">
          <Topbar
            title="New Survey"
            subtitle="Upload side-scan sonar data for AI analysis."
          />

          <section className="survey-form">
            <div className="form-card">
              <div className="section-title">
                <span className="section-icon">◈</span>

                <div>
                  <h2>Sonar Data</h2>
                  <p>
                    Select a side-scan sonar image to analyze.
                  </p>
                </div>
              </div>

              <label className="upload-area">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={(event) => {
                    setSonarFile(event.target.files?.[0] || null);
                    setError("");
                  }}
                />

                <div className="upload-icon">↑</div>

                {sonarFile ? (
                  <>
                    <strong>{sonarFile.name}</strong>
                    <span>File selected successfully</span>
                  </>
                ) : (
                  <>
                    <strong>Upload sonar image</strong>
                    <span>PNG, JPG or JPEG</span>
                  </>
                )}
              </label>
            </div>

            <div className="form-card">
              <div className="section-title">
                <span className="section-icon">⌖</span>

                <div>
                  <h2>Survey Telemetry</h2>
                  <p>
                    Enter the location and orientation of the survey.
                  </p>
                </div>
              </div>

              <div className="input-grid">
                <div className="input-group">
                  <label>Latitude</label>

                  <input
                    type="number"
                    step="any"
                    placeholder="12.2958"
                    value={latitude}
                    onChange={(event) =>
                      setLatitude(event.target.value)
                    }
                  />
                </div>

                <div className="input-group">
                  <label>Longitude</label>

                  <input
                    type="number"
                    step="any"
                    placeholder="76.6394"
                    value={longitude}
                    onChange={(event) =>
                      setLongitude(event.target.value)
                    }
                  />
                </div>

                <div className="input-group">
                  <label>Heading</label>

                  <input
                    type="number"
                    step="any"
                    placeholder="0"
                    value={heading}
                    onChange={(event) =>
                      setHeading(event.target.value)
                    }
                  />
                </div>
              </div>
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            <div className="survey-actions">
              <button
                className="secondary-button"
                onClick={() => setPage("dashboard")}
              >
                ← Back
              </button>

              <button
                className="primary-button"
                disabled={analyzing}
                onClick={analyzeSonar}
              >
                {analyzing
                  ? "⏳ Analyzing..."
                  : "🔍 Analyze Sonar"}
              </button>
            </div>
          </section>
        </main>
      </div>
    );
  }

  if (page === "change-detection") {
  const currentDetections = analysisResult?.detections || [];

  const newTargets = currentDetections.filter(
    (target) => target.comparison_status === "NEW"
  );

  const existingTargets = currentDetections.filter(
    (target) => target.comparison_status === "EXISTING"
  );

  const reviewTargets = currentDetections.filter(
    (target) =>
      target.status === "FLAGGED_FOR_REVIEW" ||
      target.multi_step_verification?.gate3_status === "UNKNOWN"
  );

  return (
    <div className="app">
      <Sidebar />

      <main className="main-content">
        <Topbar
          title="Change Detection"
          subtitle="Compare current survey targets with seafloor memory"
        />

        <div className="stats">
          <div className="stat-card">
            <span>CURRENT TARGETS</span>
            <strong>{currentDetections.length}</strong>
            <small>Targets in latest survey</small>
          </div>

          <div className="stat-card">
            <span>NEW TARGETS</span>
            <strong>{newTargets.length}</strong>
            <small>Not found in previous memory</small>
          </div>

          <div className="stat-card">
            <span>EXISTING TARGETS</span>
            <strong>{existingTargets.length}</strong>
            <small>Matched with previous survey</small>
          </div>

          <div className="stat-card warning">
            <span>NEEDS REVIEW</span>
            <strong>{reviewTargets.length}</strong>
            <small>Low-confidence or uncertain</small>
          </div>
        </div>

        <div className="form-card">
          <div className="card-header">
            <div>
              <span className="eyebrow">SURVEY COMPARISON</span>
              <h2>Seafloor Change Analysis</h2>
            </div>
          </div>

          {!analysisResult ? (
            <div style={{ padding: "30px 0", opacity: 0.7 }}>
              <h3>No current survey available</h3>
              <p>
                Analyze a sonar survey first. SONARIS will then compare the
                detected targets with previously stored seafloor memory.
              </p>

              <button
                className="primary-button"
                onClick={startNewSurvey}
                style={{ marginTop: "16px" }}
              >
                ＋ Start New Survey
              </button>
            </div>
          ) : currentDetections.length === 0 ? (
            <div style={{ padding: "30px 0", opacity: 0.7 }}>
              <h3>No targets detected</h3>
              <p>
                The current survey does not contain any detected anomalies.
              </p>
            </div>
          ) : (
            <div className="memory-list">
              {currentDetections.map((target, index) => {
                const status =
                  target.change_status || target.comparison_status || "UNKNOWN";

                const statusClass =
                  status === "NEW"
                    ? "high"
                    : status === "EXISTING"
                    ? "low"
                    : "medium";

                return (
                  <div
                    className="memory-item"
                    key={`${target.target_id || "target"}-${index}`}
                  >
                    <div>
                      <strong>
                        {target.target_id || `T${String(index + 1).padStart(3, "0")}`}
                      </strong>

                      <p>
                        {formatLabel(
                          target.label ||
                            target.predicted_class ||
                            "Unknown target"
                        )}
                      </p>
                    </div>

                    <div>
                      <span className={statusClass}>
                        {status}
                      </span>
                    </div>

                    <div>
                      <span>
                        Confidence:{" "}
                        {formatConfidence(target.confidence)}
                      </span>

                      <span>
                        IoU:{" "}
                        {target.comparison_iou !== undefined
                          ? Number(target.comparison_iou).toFixed(2)
                          : "N/A"}
                      </span>
                    </div>

                    <div>
                      <span>
                        Previous Target:{" "}
                        {target.previous_target_id || "None"}
                      </span>

                      <span>
                        Previous Survey:{" "}
                        {target.previous_survey_id || "None"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
  // ------------------------------------------------------------
  // RESULTS
  // ------------------------------------------------------------
    // SEAFLOOR MEMORY
  if (page === "memory") {
    const surveyGroups = memoryRecords.reduce((groups, record) => {
      const surveyId = record.survey_id || "UNKNOWN SURVEY";

      if (!groups[surveyId]) {
        groups[surveyId] = [];
      }

      groups[surveyId].push(record);
      return groups;
    }, {});

    return (
      <div className="app">
        <Sidebar />

        <main className="main-content">
          <Topbar
            title="Seafloor Memory"
            subtitle="Persistent survey archive of previously detected underwater targets"
          />

          <div className="form-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">SEAFLOOR MEMORY</span>
                <h2>Survey Archive</h2>
              </div>

              <span className="queue-count">
                {memoryRecords.length}
              </span>
            </div>

            {memoryLoading ? (
              <p>Loading seafloor memory...</p>
            ) : memoryError ? (
              <p>{memoryError}</p>
            ) : memoryRecords.length === 0 ? (
              <p>No seafloor targets have been stored yet.</p>
            ) : (
              <div className="memory-list">
                {Object.entries(surveyGroups).map(([surveyId, targets]) => {
                  const surveyReference = targets[0] || {};

                  return (
                    <div
                      className="survey-group"
                      key={surveyId}
                      style={{
                        marginBottom: "28px",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "14px",
                        overflow: "hidden",
                        background: "rgba(255,255,255,0.02)",
                      }}
                    >
                      <div
                        className="survey-header"
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          gap: "20px",
                          padding: "20px 22px",
                          borderBottom: "1px solid rgba(255,255,255,0.08)",
                        }}
                      >
                        <div>
                          <span className="eyebrow">SURVEY</span>
                          <h3 style={{ margin: "6px 0 8px" }}>{surveyId}</h3>

                          <div
                            style={{
                              display: "flex",
                              flexWrap: "wrap",
                              gap: "14px 24px",
                              opacity: 0.72,
                              fontSize: "13px",
                            }}
                          >
                            <span>
                              Location: {surveyReference.latitude ?? "N/A"},{" "}
                              {surveyReference.longitude ?? "N/A"}
                            </span>

                            <span>
                              Heading: {surveyReference.heading ?? "N/A"}°
                            </span>
                          </div>
                        </div>

                        <span className="queue-count">
                          {targets.length}{" "}
                          {targets.length === 1 ? "TARGET" : "TARGETS"}
                        </span>
                      </div>

                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: "1px",
                        }}
                      >
                        {targets.map((record, index) => {
                          const verificationStatus =
                            record.verification_status ||
                            record.multi_step_verification?.gate3_status ||
                            "N/A";

                          const changeStatus =
                            record.change_status ||
                            record.comparison_status ||
                            "N/A";

                          const shadowConfirmed =
                            record.shadow_confirmed ??
                            record.multi_step_verification?.gate2_shadow_confirmed ??
                            false;

                          return (
                            <div
                              className="memory-item"
                              key={`${surveyId}-${record.target_id || "target"}-${index}`}
                              style={{ borderRadius: 0 }}
                            >
                              <div>
                                <span className="eyebrow">TARGET</span>
                                <strong>
                                  {record.target_id || `T${index + 1}`}
                                </strong>

                                <p>
                                  {formatLabel(
                                    record.predicted_class ||
                                      record.label ||
                                      "Unknown target"
                                  )}
                                </p>
                              </div>

                              <div>
                                <span>
                                  Confidence:{" "}
                                  {formatConfidence(record.confidence)}
                                </span>

                                <span>
                                  Verification: {verificationStatus}
                                </span>

                                <span>
                                  Change: {changeStatus}
                                </span>
                              </div>

                              <div>
                                <span>
                                  Shadow:{" "}
                                  {shadowConfirmed
                                    ? "CONFIRMED"
                                    : "NOT CONFIRMED"}
                                </span>

                                <span>
                                  Shadow Score:{" "}
                                  {record.shadow_score ?? "N/A"}
                                </span>

                                <span>
                                  IoU:{" "}
                                  {record.comparison_iou !== undefined
                                    ? Number(record.comparison_iou).toFixed(2)
                                    : "N/A"}
                                </span>
                              </div>

                              <div>
                                <span>
                                  Location: {record.latitude ?? "N/A"},{" "}
                                  {record.longitude ?? "N/A"}
                                </span>

                                <span>
                                  Heading: {record.heading ?? "N/A"}°
                                </span>

                                <span>
                                  Previous Target:{" "}
                                  {record.previous_target_id || "None"}
                                </span>

                                <span>
                                  Previous Survey:{" "}
                                  {record.previous_survey_id || "None"}
                                </span>
                              </div>

                              <div
                                style={{
                                  gridColumn: "1 / -1",
                                  paddingTop: "12px",
                                  marginTop: "2px",
                                  borderTop:
                                    "1px solid rgba(255,255,255,0.06)",
                                  display: "grid",
                                  gridTemplateColumns:
                                    "minmax(0, 1fr) minmax(0, 1fr)",
                                  gap: "10px 24px",
                                }}
                              >
                                <span>
                                  Bounding Box:{" "}
                                  {Array.isArray(record.bbox)
                                    ? `[${record.bbox.join(", ")}]`
                                    : "N/A"}
                                </span>

                                <span>
                                  Detected:{" "}
                                  {record.timestamp
                                    ? new Date(
                                        record.timestamp
                                      ).toLocaleString()
                                    : "N/A"}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </main>
      </div>
    );
  }

  if (page === "results") {
    if (!analysisResult) {
      return (
        <div className="app">
          <Sidebar />

          <main className="main-content">
            <Topbar
              title="AI Analysis Results"
              subtitle="No analysis is available yet."
            />

            <div className="form-card">
              <p>
                Run a sonar analysis first to populate the
                target intelligence view.
              </p>

              <button
                className="primary-button"
                onClick={startNewSurvey}
              >
                ＋ Start New Survey
              </button>
            </div>
          </main>
        </div>
      );
    }

    const telemetry = analysisResult.telemetry || {};
    const visualization =
      analysisResult.visual_render_base64 ||
      analysisResult.visual_render_url ||
      null;

    return (
      <div className="app">
        <Sidebar />

        <main className="main-content">
          <Topbar
            title="AI Analysis Results"
            subtitle="SONARIS automated underwater anomaly detection report."
            status="Analysis Complete"
          />

          <section className="dashboard-grid">
            <div className="analysis-card">
              <div className="card-header">
                <div>
                  <span className="eyebrow">
                    SIDE-SCAN SONAR
                  </span>

                  <h2>
                    {analysisResult.filename ||
                      sonarFile?.name ||
                      "Survey Image"}
                  </h2>
                </div>

                <span className="live-badge">
                  LIVE RESULT
                </span>
              </div>

              <div className="sonar-view">
                {visualization ? (
                  <img
                    src={visualization}
                    alt="SONARIS AI sonar analysis"
                  />
                ) : sonarFile ? (
                  <img
                    src={URL.createObjectURL(sonarFile)}
                    alt="Uploaded sonar"
                  />
                ) : (
                  <div
                    style={{
                      minHeight: "320px",
                      display: "grid",
                      placeItems: "center",
                      opacity: 0.65,
                    }}
                  >
                    AI visualization unavailable
                  </div>
                )}
              </div>

              <div className="survey-meta">
                <span>
                  LAT{" "}
                  {telemetry.latitude ??
                    telemetry.lat ??
                    "—"}
                </span>

                <span>
                  LON{" "}
                  {telemetry.longitude ??
                    telemetry.lon ??
                    "—"}
                </span>

                <span>
                  HDG{" "}
                  {telemetry.heading ?? "—"}°
                </span>
              </div>
            </div>

{/* Sonar Quality & Preprocessing */}
<div className="analysis-card preprocessing-card" style={{ marginTop: "16px" }}>
  <div className="card-header">
    <div>
      <span className="eyebrow">SONAR PROCESSING</span>
      <h3>Quality & Preprocessing</h3>
    </div>
  </div>

  {analysisResult.preprocessing_info ? (
    <>
      <div className="target-detail-grid">
        <div className="detail-box">
          <span>BRIGHTNESS</span>
          <strong>
            {analysisResult.preprocessing_info.quality?.brightness ?? "N/A"}
          </strong>
        </div>

        <div className="detail-box">
          <span>CONTRAST</span>
          <strong>
            {analysisResult.preprocessing_info.quality?.contrast ?? "N/A"}
          </strong>
        </div>

        <div className="detail-box">
          <span>NOISE ESTIMATE</span>
          <strong>
            {analysisResult.preprocessing_info.quality?.noise_estimate ?? "N/A"}
          </strong>
        </div>

        <div className="detail-box">
          <span>PREPROCESSING STEP</span>
          <strong>
            {analysisResult.preprocessing_info.steps?.join(", ") || "None"}
          </strong>
        </div>
      </div>
    </>
  ) : (
    <p>Preprocessing information unavailable.</p>
  )}
</div>

<div className="priority-card">
              <div className="card-header">
                <div>
                  <span className="eyebrow">
                    DETECTION QUEUE
                  </span>

                  <h2>Target Intelligence</h2>
                </div>

                <span className="queue-count">
                  {analysisResult.total_targets ??
                    detections.length}
                </span>
              </div>

              <TargetList items={detections} />

              {selectedTarget && (
                <TargetDetails
                  target={selectedTarget}
                  index={detections.findIndex((target) =>
                    sameTarget(target, selectedTarget)
                  )}
                />
              )}

              <button
                className="primary-button full"
                onClick={startNewSurvey}
              >
                ＋ Start New Survey
              </button>
            </div>
          </section>

          <div
            style={{
              display: "flex",
              gap: "12px",
              marginTop: "20px",
            }}
          >
            <button
              className="secondary-button"
              onClick={() => setPage("dashboard")}
            >
              ← Dashboard
            </button>

            <button
              className="primary-button"
              onClick={startNewSurvey}
            >
              Analyze Another Image
            </button>
          </div>
        </main>
      </div>
    );
  }

  // ------------------------------------------------------------
  // DASHBOARD
  // ------------------------------------------------------------
  const totalTargets =
    analysisResult?.total_targets ?? detections.length;

  const highRiskTargets = detections.filter(
    (target) => getRisk(target) >= 70
  ).length;

  const verifiedTargets = detections.filter(
    (target) =>
      getVerification(target).status === "VERIFIED"
  ).length;

  const telemetry = analysisResult?.telemetry || {};

  return (
    <div className="app">
      <Sidebar />

      <main className="main-content">
        <Topbar
          title="Mission Dashboard"
          subtitle="Seafloor Analysis • AI-powered underwater intelligence"
        />

        <section className="stats">
          <div className="stat-card">
            <span>DETECTED TARGETS</span>
            <strong>{totalTargets}</strong>
            <small>
              {analysisResult
                ? "Latest AI analysis"
                : "Awaiting survey"}
            </small>
          </div>

          <div className="stat-card warning">
            <span>HIGH RISK</span>
            <strong>{highRiskTargets}</strong>
            <small>Requires operator review</small>
          </div>

          <div className="stat-card">
            <span>VERIFIED</span>
            <strong>{verifiedTargets}</strong>
            <small>Multi-evidence validation</small>
          </div>

          <div className="stat-card">
            <span>SYSTEM</span>
            <strong>ONLINE</strong>
            <small>AI Engine Ready</small>
          </div>
        </section>

        <section className="dashboard-grid">
          <div className="analysis-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">
                  SEAFLOOR ANALYSIS
                </span>

                <h2>SIDE-SCAN SONAR</h2>
              </div>

              <span className="live-badge">
                {analysisResult
                  ? "ANALYSIS READY"
                  : "LIVE"}
              </span>
            </div>

            <div className="sonar-view">
              {analysisResult?.visual_render_base64 ? (
                <img
                  src={analysisResult.visual_render_base64}
                  alt="SONARIS analysis"
                />
              ) : analysisResult?.visual_render_url ? (
                <img
                  src={analysisResult.visual_render_url}
                  alt="SONARIS analysis"
                />
              ) : (
                <div
                  style={{
                    minHeight: "320px",
                    display: "grid",
                    placeItems: "center",
                    textAlign: "center",
                    padding: "40px",
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontSize: "42px",
                        marginBottom: "12px",
                      }}
                    >
                      ◈
                    </div>

                    <strong>
                      No survey loaded
                    </strong>

                    <p style={{ opacity: 0.65 }}>
                      Upload a side-scan sonar image
                      to begin AI analysis.
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="survey-meta">
              <span>
                LOCATION{" "}
                {telemetry.latitude ??
                  telemetry.lat ??
                  "12.2958"}{" "}
                N,{" "}
                {telemetry.longitude ??
                  telemetry.lon ??
                  "76.6394"}{" "}
                E
              </span>

              <span>
                HEADING{" "}
                {telemetry.heading ?? "084"}°
              </span>
            </div>
          </div>

          <div className="priority-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">
                  PRIORITY QUEUE
                </span>

                <h2>Detected Targets</h2>
              </div>

              <span className="queue-count">
                {totalTargets}
              </span>
            </div>

            <TargetList items={detections} />

            {selectedTarget && (
              <TargetDetails
                target={selectedTarget}
                index={detections.findIndex((target) =>
                  sameTarget(target, selectedTarget)
                )}
              />
            )}

            <button
              className="primary-button full"
              onClick={
                analysisResult
                  ? () => setPage("results")
                  : startNewSurvey
              }
            >
              {analysisResult
                ? "View Full Analysis"
                : "＋ Start New Survey"}
            </button>
          </div>
        </section>

        {analysisResult && (
          <div
            style={{
              display: "flex",
              gap: "12px",
              marginTop: "20px",
            }}
          >
            <button
              className="secondary-button"
              onClick={() => setPage("results")}
            >
              ◈ View Targets
            </button>

            <button
              className="primary-button"
              onClick={startNewSurvey}
            >
              ＋ New Survey
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
