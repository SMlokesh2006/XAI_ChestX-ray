import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, EyeOff, Info } from 'lucide-react';
import { twMerge } from 'tailwind-merge';

export function GradCAMView({ originalImage, heatmapImage, heatmapOnly, explanation }) {
    const [showHeatmap, setShowHeatmap] = useState(true);

    if (!heatmapImage) return null;

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 flex flex-col gap-6"
        >
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                    Explainability Analysis (Grad-CAM++)
                </h3>
                <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-slate-200 text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors shadow-sm"
                >
                    {showHeatmap ? <EyeOff size={16} /> : <Eye size={16} />}
                    {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Visualization Area */}
                <div className="relative aspect-square rounded-2xl overflow-hidden shadow-lg border border-slate-200 bg-slate-900 group">
                    {/* Base Image */}
                    <img
                        src={showHeatmap ? `data:image/png;base64,${heatmapImage}` : URL.createObjectURL(originalImage)}
                        alt="Analysis"
                        className="w-full h-full object-contain transition-all duration-300"
                    />

                    <div className="absolute top-4 left-4">
                        <span className="bg-black/60 backdrop-blur-md text-white text-xs px-2 py-1 rounded-md border border-white/10">
                            {showHeatmap ? 'Original + Heatmap' : 'Original X-ray'}
                        </span>
                    </div>

                    {!showHeatmap && (
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                            <span className="bg-black/70 text-white px-3 py-1 rounded-full text-sm">Click 'Show Heatmap' to see AI focus</span>
                        </div>
                    )}
                </div>

                {/* Legend & Explanation */}
                <div className="flex flex-col gap-4">
                    <div className="flex-1 bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                        <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                            <Info size={16} className="text-medical-500" />
                            AI Reasoning
                        </h4>
                        <p className="text-slate-600 text-sm leading-relaxed mb-4">
                            {explanation || "The model analyzes pixel intensity patterns to identify potential anomalies associated with pneumonia."}
                        </p>

                        <div className="border-t border-slate-100 pt-4">
                            <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Heatmap Legend</h5>
                            <div className="flex items-center gap-3 text-xs">
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full bg-red-600" />
                                    <span className="text-slate-600">High Attention</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full bg-yellow-400" />
                                    <span className="text-slate-600">Medium</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                                    <span className="text-slate-600">Low</span>
                                </div>
                            </div>
                            <p className="text-xs text-slate-400 mt-2">
                                Red regions indicate the areas that most influenced the model's decision.
                            </p>
                        </div>
                    </div>

                    {/* Mini Heatmap Only View */}
                    {heatmapOnly && (
                        <div className="h-32 bg-slate-100 rounded-xl overflow-hidden relative border border-slate-200">
                            <img src={`data:image/png;base64,${heatmapOnly}`} className="w-full h-full object-cover opacity-80" alt="Heatmap Only" />
                            <div className="absolute bottom-2 right-2 text-[10px] bg-white/80 px-1.5 rounded text-slate-600">
                                Raw Activation
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}
