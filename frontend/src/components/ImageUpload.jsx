import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage, X } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function ImageUpload({ onFileSelect, selectedFile, onClear }) {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles?.length > 0) {
            onFileSelect(acceptedFiles[0]);
        }
    }, [onFileSelect]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png']
        },
        multiple: false,
        disabled: !!selectedFile
    });

    if (selectedFile) {
        return (
            <div className="relative w-full h-80 rounded-2xl overflow-hidden bg-slate-900 shadow-lg border border-slate-700/50 group">
                <img
                    src={URL.createObjectURL(selectedFile)}
                    alt="Preview"
                    className="w-full h-full object-contain p-4"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button
                        onClick={onClear}
                        className="bg-white/10 hover:bg-white/20 backdrop-blur-md text-white px-4 py-2 rounded-full flex items-center gap-2 transition-colors border border-white/20"
                    >
                        <X size={18} />
                        Remove Image
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div
            {...getRootProps()}
            className={twMerge(
                "w-full h-80 rounded-2xl border-2 border-dashed transition-all duration-200 flex flex-col items-center justify-center cursor-pointer relative overflow-hidden group",
                isDragActive
                    ? "border-medical-400 bg-medical-50/10"
                    : "border-slate-300 hover:border-medical-300 hover:bg-slate-50"
            )}
        >
            <input {...getInputProps()} />

            <div className="absolute inset-0 bg-gradient-to-tr from-medical-50/0 to-medical-50/0 group-hover:from-medical-50/30 group-hover:to-transparent transition-all duration-500" />

            <div className="z-10 bg-white p-4 rounded-full shadow-lg text-medical-500 mb-4 group-hover:scale-110 transition-transform duration-300">
                <Upload size={32} />
            </div>

            <h3 className="text-lg font-semibold text-slate-700 z-10 transition-colors group-hover:text-medical-700">
                Upload Chest X-ray
            </h3>

            <p className="text-sm text-slate-500 mt-2 z-10 max-w-[200px] text-center">
                Drag and drop your image here, or click to browse
            </p>

            <div className="mt-4 flex items-center gap-2 text-xs text-slate-400 z-10">
                <FileImage size={14} />
                <span>Supports JPG, PNG</span>
            </div>
        </div>
    );
}
