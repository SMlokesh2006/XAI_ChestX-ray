import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, AlertTriangle, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function PredictionCard({ prediction, confidence, loading }) {
    if (loading) {
        return (
            <div className="w-full p-6 rounded-2xl bg-white shadow-sm border border-slate-200 animate-pulse">
                <div className="h-6 w-32 bg-slate-200 rounded mb-4" />
                <div className="h-10 w-full bg-slate-200 rounded mb-4" />
                <div className="h-4 w-2/3 bg-slate-200 rounded" />
            </div>
        );
    }

    if (!prediction) return null;

    const isNormal = prediction === 'NORMAL';
    const colorClass = isNormal ? 'text-emerald-600' : 'text-rose-600';
    const bgClass = isNormal ? 'bg-emerald-50' : 'bg-rose-50';
    const borderClass = isNormal ? 'border-emerald-200' : 'border-rose-200';
    const Icon = isNormal ? CheckCircle2 : AlertTriangle;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={twMerge(
                "w-full p-6 rounded-2xl shadow-sm border relative overflow-hidden",
                bgClass, borderClass
            )}
        >
            <div className="flex items-start justify-between relative z-10">
                <div>
                    <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-2">
                        <Activity size={16} />
                        Prediction Result (ResNet18)
                    </h4>
                    <div className="flex items-center gap-3">
                        <Icon size={32} className={colorClass} />
                        <span className={clsx("text-3xl font-bold tracking-tight", colorClass)}>
                            {prediction}
                        </span>
                    </div>
                </div>

                <div className="text-right">
                    <span className="block text-sm font-medium text-slate-500 mb-1">Confidence</span>
                    <span className={clsx("text-2xl font-bold", colorClass)}>
                        {confidence.toFixed(1)}%
                    </span>
                </div>
            </div>

            <div className="mt-6 relative z-10">
                <div className="h-3 w-full bg-slate-200/50 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className={clsx("h-full rounded-full", isNormal ? 'bg-emerald-500' : 'bg-rose-500')}
                    />
                </div>
            </div>

            {/* Background Decor */}
            <Icon className={clsx("absolute -right-6 -bottom-6 w-32 h-32 opacity-5", colorClass)} />
        </motion.div>
    );
}
