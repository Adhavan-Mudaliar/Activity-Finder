import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Search, Play, Clock, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [videoFile, setVideoFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, processing, success, error
  const [videoId, setVideoId] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [processedVideos, setProcessedVideos] = useState([]);
  const [videoUrl, setVideoUrl] = useState('');
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const res = await axios.get('/api/videos');
      setProcessedVideos(res.data.videos || []);
    } catch (err) {
      console.error("Failed to fetch videos", err);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!videoFile) return;

    const formData = new FormData();
    formData.append('file', videoFile);

    setUploadStatus('uploading');
    try {
      const res = await axios.post('/api/upload_video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setVideoId(res.data.video_id);
      setUploadStatus('success');
      fetchVideos();
      setActiveTab('search');
    } catch (err) {
      setUploadStatus('error');
      console.error(err);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!videoId || !searchQuery) return;

    setIsSearching(true);
    try {
      const res = await axios.post('/api/search_scene', {
        video_id: videoId,
        query: searchQuery,
        k: 5
      });
      setSearchResults(res.data.scenes || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const jumpToScene = (seconds) => {
    const video = document.getElementById('main-video');
    if (video) {
      video.currentTime = seconds;
      video.play();
    }
  };

  return (
    <div className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="flex justify-between items-center mb-12">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          Video Scene Retrieval
        </h1>
        <nav className="flex gap-4">
          <button 
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-2 rounded-full transition ${activeTab === 'upload' ? 'bg-white/10' : 'hover:bg-white/5'}`}
          >
            Upload
          </button>
          <button 
            onClick={() => setActiveTab('search')}
            className={`px-4 py-2 rounded-full transition ${activeTab === 'search' ? 'bg-white/10' : 'hover:bg-white/5'}`}
          >
            Search
          </button>
        </nav>
      </header>

      <main>
        {activeTab === 'upload' ? (
          <section className="glass p-10 text-center">
            <div className="max-w-md mx-auto">
              <Upload className="w-16 h-16 mx-auto mb-6 text-indigo-400" />
              <h2 className="text-2xl font-semibold mb-4">Upload Video</h2>
              <p className="text-gray-400 mb-8">Process your video to extract searchable scenes using AI.</p>
              
              <form onSubmit={handleUpload}>
                <div className="border-2 border-dashed border-white/10 rounded-2xl p-8 mb-6 hover:border-indigo-400 transition cursor-pointer relative">
                  <input 
                    type="file" 
                    onChange={(e) => setVideoFile(e.target.files[0])}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    accept="video/*"
                  />
                  {videoFile ? (
                    <div className="flex items-center justify-center gap-2">
                      <CheckCircle className="text-green-400" />
                      <span>{videoFile.name}</span>
                    </div>
                  ) : (
                    <span>Click or drag video file here</span>
                  )}
                </div>
                
                <button 
                  type="submit" 
                  disabled={!videoFile || uploadStatus === 'uploading'}
                  className="primary-btn w-full flex items-center justify-center gap-2"
                >
                  {uploadStatus === 'uploading' ? (
                    <><Loader2 className="animate-spin" /> Processing...</>
                  ) : (
                    'Start Pipeline'
                  )}
                </button>
              </form>
              
              {uploadStatus === 'success' && (
                <div className="mt-6 flex items-center justify-center gap-2 text-green-400">
                  <CheckCircle size={20} />
                  <span>Video processed successfully! ID: {videoId}</span>
                </div>
              )}
              {uploadStatus === 'error' && (
                <div className="mt-6 flex items-center justify-center gap-2 text-red-400">
                  <AlertCircle size={20} />
                  <span>Upload or processing failed.</span>
                </div>
              )}
            </div>
          </section>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <section className="lg:col-span-2 space-y-8">
              <div className="glass p-6">
                <form onSubmit={handleSearch} className="flex gap-4">
                  <div className="relative flex-1">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
                    <input 
                      type="text" 
                      placeholder="Search for a scene (e.g., 'a person waving')"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-12"
                    />
                  </div>
                  <button type="submit" className="primary-btn" disabled={isSearching}>
                    {isSearching ? <Loader2 className="animate-spin" /> : 'Search'}
                  </button>
                </form>
              </div>

              {videoId && (
                <div className="glass overflow-hidden">
                  <video 
                    id="main-video"
                    controls 
                    className="w-full aspect-video bg-black"
                    src={`/api/video/${videoId}/stream`}
                  />
                </div>
              )}

              <div className="space-y-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <Play size={20} className="text-indigo-400" />
                  Search Results
                </h3>
                {searchResults.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {searchResults.map((scene, i) => (
                      <div key={i} className="glass p-4 hover:bg-white/5 transition group">
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-sm font-mono text-indigo-300 bg-indigo-500/10 px-2 py-1 rounded">
                            {scene.start_time} - {scene.end_time}
                          </span>
                          <span className="text-xs text-gray-500">
                            Score: {(scene.confidence_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <button 
                          onClick={() => jumpToScene(scene.start_seconds)}
                          className="w-full mt-4 py-2 text-sm border border-white/10 rounded-lg hover:bg-white/10 transition flex items-center justify-center gap-2"
                        >
                          <Play size={14} fill="currentColor" />
                          Jump to Scene
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="glass p-12 text-center text-gray-500">
                    {isSearching ? 'Analyzing video frames...' : 'No results yet. Try a different query.'}
                  </div>
                )}
              </div>
            </section>

            <aside className="space-y-6">
               <div className="glass p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Clock size={18} className="text-indigo-400" />
                    Videos
                  </h3>
                  <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                    {processedVideos.length > 0 ? (
                      processedVideos.map((v) => (
                        <div 
                          key={v.video_id}
                          onClick={() => setVideoId(v.video_id)}
                          className={`p-3 rounded-xl cursor-pointer transition ${videoId === v.video_id ? 'bg-indigo-500/20 border-indigo-500/30 border' : 'hover:bg-white/5'}`}
                        >
                          <div className="text-sm font-medium truncate">{v.video_id}</div>
                          <div className="text-xs text-gray-500 mt-1">{v.duration} • {v.frame_count} frames</div>
                        </div>
                      ))
                    ) : (
                      <div className="text-sm text-gray-500">No videos processed yet.</div>
                    )}
                  </div>
               </div>
            </aside>
          </div>
        )}
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
        .grid-cols-1 md:grid-cols-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
        .min-h-screen { min-height: 100vh; }
        .bg-clip-text { -webkit-background-clip: text; background-clip: text; }
        .animate-spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default App;
