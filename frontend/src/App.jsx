import React, { useState } from 'react';
import axios from 'axios';
import { Activity } from 'lucide-react';
import { Header } from './components/Header';
import { ImageUpload } from './components/ImageUpload';
import { PredictionCard } from './components/PredictionCard';
import { GradCAMView } from './components/GradCAMView';

// Use environment variable or default to localhost
const API_URL = 'http://localhost:5000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      console.error(err);
      setError('Failed to analyze image. Please ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      <Header />

      <main className="pt-24 px-6 lg:px-12 max-w-7xl mx-auto">

        {/* Intro */}
        <div className="mb-10 text-center max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold text-slate-800 tracking-tight mb-3">
            Pneumonia Detection & Explanation
          </h2>
          <p className="text-slate-500">
            Upload a chest X-ray image to detect pneumonia and visualize the AI's reasoning using Grad-CAM++ technology.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

          {/* Left Panel: Upload */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100">
              <h3 className="text-lg font-semibold mb-4 text-slate-700">Input Image</h3>
              <ImageUpload
                selectedFile={selectedFile}
                onFileSelect={handleFileSelect}
                onClear={handleClear}
              />

              <div className="mt-6">
                <button
                  onClick={handleAnalyze}
                  disabled={!selectedFile || loading}
                  className="w-full py-3.5 px-6 bg-medical-500 hover:bg-medical-600 active:bg-medical-700 text-white rounded-xl font-medium shadow-lg shadow-medical-500/25 transition-all disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    'Analyze X-Ray'
                  )}
                </button>
              </div>

              {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg border border-red-100">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: Results */}
          <div className="lg:col-span-7 space-y-6">
            {/* Placeholder when empty */}
            {!prediction && !loading && !error && (
              <div className="h-full min-h-[400px] flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-200 rounded-3xl bg-slate-50/50">
                <Activity size={48} className="mb-4 opacity-50" />
                <p>Analysis results will appear here</p>
              </div>
            )}

            {(prediction || loading) && (
              <div className="space-y-6">
                <PredictionCard
                  prediction={prediction?.prediction}
                  confidence={prediction?.confidence}
                  loading={loading}
                />

                {prediction && (
                  <GradCAMView
                    originalImage={selectedFile}
                    heatmapImage={prediction.heatmap}
                    heatmapOnly={prediction.heatmap_only}
                    explanation={prediction.explanation}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
