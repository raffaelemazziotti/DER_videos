// Load JSON with video info and build dropdown browser
fetch("video_info.json")
    .then(response => response.json())
    .then(data => {

        const container = document.getElementById("dataset-container");
        container.innerHTML = ""; // remove 'Loading dataset...'

        // Dropdown
        const label = document.createElement("label");
        label.setAttribute("for", "recording-select");
        label.textContent = "Select recording:";

        const select = document.createElement("select");
        select.id = "recording-select";

        const detailsDiv = document.createElement("div");
        detailsDiv.id = "recording-details";

        container.appendChild(label);
        container.appendChild(select);
        container.appendChild(detailsDiv);

        // Populate dropdown
        const keys = Object.keys(data).sort();
        keys.forEach(key => {
            const opt = document.createElement("option");
            opt.value = key;
            opt.textContent = key;
            select.appendChild(opt);
        });

        // Rendering function
        function renderRecording(recKey) {
            const info = data[recKey];
            if (!info) {
                detailsDiv.innerHTML = "<p>No information available for this recording.</p>";
                return;
            }

            const metaHTML = `
                <table>
                    <tr><td>FPS</td><td>${info.fps}</td></tr>
                    <tr><td>Total frames</td><td>${info.total_frames}</td></tr>
                    <tr><td>Resolution</td><td>${info.resolution[0]} x ${info.resolution[1]}</td></tr>
                    <tr><td>Duration (min)</td><td>${info.duration_min}</td></tr>
                    <tr><td>Num trials</td><td>${info.num_trials}</td></tr>
                </table>
            `;

            const gifPath = `GIFs/${recKey}.gif`;
            const svgPath = `graphs/${recKey}_movement_matrix.svg`;

            detailsDiv.innerHTML = `
                ${metaHTML}
                <div class="preview-row">

                    <div class="preview-item1">
                        <h3>Preview GIF</h3>
                        <div class="preview-box">
                            <img src="${gifPath}" alt="Preview GIF">
                        </div>
                    </div>

                    <div class="preview-item2">
                        <h3>Movement Matrix</h3>
                        <div class="preview-box">
                            <img src="${svgPath}" alt="Movement Matrix">
                        </div>
                    </div>

                </div>
            `;
        }

        // React to selection
        select.addEventListener("change", e => {
            renderRecording(e.target.value);
        });

        // Show first entry by default
        if (keys.length > 0) {
            renderRecording(keys[0]);
        }
    })
    .catch(err => {
        document.getElementById("dataset-container").innerHTML =
            "<p>Error loading dataset JSON.</p>";
        console.error(err);
    });


// Email obfuscation
document.addEventListener("DOMContentLoaded", () => {
    const p1 = "raffaelemario";
    const p2 = "mazziotti";
    const p3 = "unifi";
    const p4 = "it";

    const email = `${p1}.${p2}@${p3}.${p4}`;

    const contactElement = document.getElementById("contact");
    if (contactElement) {
        contactElement.innerHTML = `For info: <a href="mailto:${email}">${email}</a>`;
    }
});
