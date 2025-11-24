// ----------------------------------------------------------
//  Contact line in footer (email obfuscation)
// ----------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    setContact();
    initDatasetBrowser();
});

/**
 * Build the "Contact" line with an obfuscated email.
 */
function setContact() {
    const p1 = "raffaelemario";
    const p2 = "mazziotti";
    const p3 = "unifi";
    const p4 = "it";

    const email = `${p1}.${p2}@${p3}.${p4}`;
    const contactElement = document.getElementById("contact");

    if (contactElement) {
        contactElement.innerHTML =
            `For info: <a href="mailto:${email}">${email}</a>`;
    }
}

// ----------------------------------------------------------
//  Dataset Browser
// ----------------------------------------------------------

/**
 * Initialize the dataset browser:
 * fetch video_info.json, then build the UI.
 */
function initDatasetBrowser() {
    const container = document.getElementById("dataset-container");
    if (!container) return;

    fetch("video_info.json")
        .then((resp) => {
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            return resp.json();
        })
        .then((data) => {
            buildDatasetUI(container, data);
        })
        .catch((err) => {
            console.error("Failed to load video_info.json:", err);
            container.textContent =
                "Could not load video_info.json. Make sure the file is in the same folder as index.html.";
        });
}

/**
 * Build the UI: dropdown, metadata table, and preview panels.
 *
 * @param {HTMLElement} container
 * @param {Object} metaData Parsed JSON from video_info.json
 */
function buildDatasetUI(container, metaData) {
    const recordingNames = Object.keys(metaData).sort();
    if (recordingNames.length === 0) {
        container.textContent = "No recordings found in video_info.json.";
        return;
    }

    container.innerHTML = `
        <div id="dataset-controls">
            <label for="recording-select">Select recording:</label>
            <select id="recording-select"></select>
        </div>

        <div id="recording-details"></div>

        <div class="preview-row">
            <div class="preview-item1">
                <h3>Preview video</h3>
                <div class="preview-box">
                    <p style="font-size:0.9rem; color:#666;">No recording selected.</p>
                </div>
            </div>

            <div class="preview-item2">
                <h3>Movement matrix</h3>
                <div class="preview-box">
                    <p style="font-size:0.9rem; color:#666;">No recording selected.</p>
                </div>
            </div>
        </div>
    `;

    const select = document.getElementById("recording-select");
    const detailsDiv = document.getElementById("recording-details");

    // Populate dropdown
    for (const name of recordingNames) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    }

    // Selection change handler
    select.addEventListener("change", () => {
        const rec = select.value;
        const info = metaData[rec];
        updateDetails(detailsDiv, rec, info);
        updatePreviews(rec);
    });

    // Initialize with the first recording
    const firstRec = recordingNames[0];
    select.value = firstRec;
    updateDetails(detailsDiv, firstRec, metaData[firstRec]);
    updatePreviews(firstRec);
}

/**
 * Fill the metadata table for a given recording.
 *
 * @param {HTMLElement} detailsDiv
 * @param {string} recName
 * @param {Object} info
 */
function updateDetails(detailsDiv, recName, info) {
    if (!info) {
        detailsDiv.textContent = "No metadata available for " + recName;
        return;
    }

    let intensityStr = "";
    if (info.intensity_counts && typeof info.intensity_counts === "object") {
        const parts = [];
        for (const [k, v] of Object.entries(info.intensity_counts)) {
            parts.push(`${k}: ${v}`);
        }
        intensityStr = parts.join(", ");
    }

    const rows = [
        ["Recording", recName],
        ["FPS", info.fps],
        ["Total frames", info.total_frames],
        ["Resolution", Array.isArray(info.resolution) ? info.resolution.join(" × ") : info.resolution],
        ["Duration (sec)", info.duration_sec],
        ["Duration (min)", info.duration_min],
        ["Number of trials", info.num_trials],
        ["First trigger frame", info.first_trigger_frame],
        ["Last trigger frame", info.last_trigger_frame],
        ["Intensity counts", intensityStr],
        ["Valid trials", info.valid_trials],
        ["Invalid trials", info.invalid_trials],
    ];

    let html = "<table><tbody>";
    for (const [label, value] of rows) {
        if (value === undefined || value === null || value === "") continue;
        html += `<tr><td>${label}</td><td>${value}</td></tr>`;
    }
    html += "</tbody></table>";

    detailsDiv.innerHTML = html;
}

/**
 * Update the preview video and movement matrix.
 *
 * Assumes:
 *   - MP4 preview path:  mp4/<recName>.mp4
 *   - SVG matrix path:   graphs/<recName>_movement_matrix.svg
 *
 * @param {string} recName
 */
function updatePreviews(recName) {
    const videoBox = document.querySelector(".preview-item1 .preview-box");
    const matrixBox = document.querySelector(".preview-item2 .preview-box");
    if (!videoBox || !matrixBox) return;

    const videoPath = `mp4/${recName}.mp4`;
    const svgPath = `graphs/${recName}_movement_matrix.svg`;

    // Video preview from mp4 folder
    videoBox.innerHTML = `
        <video autoplay loop muted playsinline preload="metadata"
               style="width:100%; height:100%; object-fit:contain; display:block;">
            <source src="${videoPath}" type="video/mp4">
            Your browser does not support MP4 playback.
        </video>
    `;

    // Movement matrix image
    matrixBox.innerHTML = `
        <img src="${svgPath}" alt="Movement matrix for ${recName}">
    `;
}
