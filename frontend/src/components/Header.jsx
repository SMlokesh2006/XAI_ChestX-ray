import React from 'react';
import { Stethoscope, Activity, Github } from 'lucide-react';

export function Header() {
    return (
        <header className="fixed top-0 left-0 right-0 h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 z-50 flex items-center justify-between px-6 lg:px-12">
            <div className="flex items-center gap-3">
                <div className="bg-medical-500 p-2 rounded-lg text-white shadow-lg shadow-medical-500/30">
                    <Stethoscope size={20} className="stroke-[2.5]" />
                </div>
                <div>
                    <h1 className="text-lg font-bold text-slate-800 tracking-tight leading-none">
                        PneumoScan<span className="text-medical-500">.AI</span>
                    </h1>
                    <p className="text-[10px] text-slate-500 font-medium tracking-wide">
                        XAI-POWERED DIAGNOSTICS
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-6">
                <nav className="hidden md:flex gap-6 text-sm font-medium text-slate-600">
                    <a href="#" className="hover:text-medical-600 transition-colors">Dashboard</a>
                    <a href="#" className="hover:text-medical-600 transition-colors">History</a>
                    <a href="#" className="hover:text-medical-600 transition-colors">Settings</a>
                </nav>
                <div className="w-px h-6 bg-slate-200 hidden md:block" />
                <a
                    href="https://github.com"
                    target='_blank'
                    rel="noreferrer"
                    className="text-slate-400 hover:text-slate-800 transition-colors"
                >
                    <Github size={20} />
                </a>
            </div>
        </header>
    );
}
