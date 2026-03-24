document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('video_id');
    const filename = urlParams.get('filename') || `${videoId}.mp4`;

    if (!videoId) {
        alert('No video ID provided. Redirecting to upload page.');
        window.location.href = 'index.html';
        return;
    }

    const mainVideo = document.getElementById('mainVideo');
    const videoSource = document.getElementById('videoSource');
    const currentVideoName = document.getElementById('currentVideoName');
    const queryInput = document.getElementById('queryInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsGrid = document.getElementById('resultsGrid');
    const loading = document.getElementById('loading');
    const resultCardTemplate = document.getElementById('resultCardTemplate');

    const API_URL = ''; // Relative path

    // Initialize Video
    console.log("Initializing video:", filename);
    currentVideoName.textContent = filename;
    mainVideo.src = `/videos/${filename}`;
    mainVideo.load();

    // Handle initial timestamp if provided
    const startTime = urlParams.get('t');
    if (startTime) {
        mainVideo.addEventListener('loadedmetadata', () => {
            console.log("Initial jump to time:", startTime);
            jumpToTime(parseFloat(startTime));
        }, { once: true });
    }

    async function performSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        console.log("Performing search for:", query, "in video:", videoId);
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
                    video_id: videoId,
                    top_k: 10
                }),
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            console.log("Search results received:", data.results.length);
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
            resultsGrid.innerHTML = '<p class="no-results">No matching scenes found for this query in this video.</p>';
            return;
        }

        results.forEach((result, index) => {
            const clone = resultCardTemplate.content.cloneNode(true);
            const card = clone.querySelector('.result-card');

            // Set delay for staggered animation
            card.style.animationDelay = `${index * 0.1}s`;

            const confidence = Math.round((result.score + result.alignment) * 50);
            clone.querySelector('.score-badge').textContent = `Match: ${confidence}%`;

            const timestampContainer = clone.querySelector('.timestamps');
            timestampContainer.innerHTML = '';

            // Focus on the first timestamp for the jump button, but show all
            const firstTs = result.timestamps[0];
            
            result.timestamps.forEach(ts => {
                const tag = document.createElement('span');
                tag.className = 'timestamp-tag';
                tag.style.cursor = 'pointer';
                tag.textContent = `${formatTime(ts[0])} - ${formatTime(ts[1])}`;
                
                // Clicking the tag itself also jumps
                tag.addEventListener('click', () => {
                    console.log("Timestamp clicked:", ts[0]);
                    jumpToTime(ts[0]);
                });
                
                timestampContainer.appendChild(tag);
            });

            const playBtn = clone.querySelector('.play-btn');
            if (firstTs) {
                playBtn.addEventListener('click', () => {
                    console.log("Jump button clicked:", firstTs[0]);
                    jumpToTime(firstTs[0]);
                });
            } else {
                playBtn.disabled = true;
                playBtn.textContent = 'No Timestamp';
            }

            resultsGrid.appendChild(clone);
        });
    }

    function jumpToTime(seconds) {
        console.log(`Setting video time to ${seconds}s`);
        if (isNaN(seconds)) return;
        
        mainVideo.currentTime = seconds;
        mainVideo.play().catch(e => {
            console.warn("Video play failed (possibly blocked):", e);
        });
        
        // Scroll video into view
        mainVideo.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
