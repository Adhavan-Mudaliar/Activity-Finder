document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('queryInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsGrid = document.getElementById('resultsGrid');
    const loading = document.getElementById('loading');
    const resultCardTemplate = document.getElementById('resultCardTemplate');

    // Upload Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const statusText = document.getElementById('statusText');

    const API_URL = ''; // Relative path since frontend is served by the API

    // --- Upload Logic ---
    browseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });

    async function handleFileUpload(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        uploadStatus.classList.remove('hidden');
        dropZone.classList.add('hidden');
        statusText.textContent = 'Uploading video...';

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            statusText.textContent = 'Video uploaded successfully! Redirecting to search page...';

            setTimeout(() => {
                window.location.href = `search.html?video_id=${data.video_id}&filename=${data.filename}`;
            }, 2000);

        } catch (error) {
            console.error('Upload Error:', error);
            statusText.textContent = 'Error during upload. Please try again.';
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
                dropZone.classList.remove('hidden');
            }, 3000);
        }
    }

    // --- Search Logic ---
    async function performSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset UI
        resultsGrid.innerHTML = '';
        loading.classList.remove('hidden');
        searchBtn.disabled = true;

        try {
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: 6
                }),
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            displayResults(data.results);
        } catch (error) {
            console.error('Error:', error);
            resultsGrid.innerHTML = `<p class="error">Error connecting to retrieval engine. Make sure the API is running.</p>`;
        } finally {
            loading.classList.add('hidden');
            searchBtn.disabled = false;
        }
    }

    function displayResults(results) {
        if (!results || results.length === 0) {
            resultsGrid.innerHTML = '<p class="no-results">No matching scenes found.</p>';
            return;
        }

        results.forEach((result, index) => {
            const clone = resultCardTemplate.content.cloneNode(true);
            const card = clone.querySelector('.result-card');

            // Set delay for staggered animation
            card.style.animationDelay = `${index * 0.1}s`;

            clone.querySelector('.video-id').textContent = `Video: ${result.video_id}`;
            clone.querySelector('.score-badge').textContent = `Match: ${Math.round((result.score + result.alignment) * 50)}%`;

            const timestampContainer = clone.querySelector('.timestamps');
            timestampContainer.innerHTML = '';

            const firstTs = result.timestamps[0];

            result.timestamps.forEach(ts => {
                const tag = document.createElement('span');
                tag.className = 'timestamp-tag';
                tag.style.cursor = 'pointer';
                tag.textContent = `${formatTime(ts[0])} - ${formatTime(ts[1])}`;
                
                tag.addEventListener('click', () => {
                    const filename = result.video_id.includes('.') ? result.video_id : `${result.video_id}.mp4`;
                    window.location.href = `search.html?video_id=${result.video_id}&filename=${filename}&t=${ts[0]}`;
                });
                
                timestampContainer.appendChild(tag);
            });

            const playBtn = clone.querySelector('.play-btn');
            if (firstTs) {
                playBtn.addEventListener('click', () => {
                    const filename = result.video_id.includes('.') ? result.video_id : `${result.video_id}.mp4`;
                    window.location.href = `search.html?video_id=${result.video_id}&filename=${filename}&t=${firstTs[0]}`;
                });
            } else {
                playBtn.disabled = true;
                playBtn.textContent = 'No Timestamp';
            }

            resultsGrid.appendChild(clone);
        });
    }

    function formatTime(seconds) {
        const date = new Date(0);
        date.setSeconds(seconds);
        return date.toISOString().substr(14, 5);
    }

    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
});
