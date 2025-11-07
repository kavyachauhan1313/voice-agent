// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD 2-Clause License

import { useRef } from "react";

interface UploadedPrompt {
  id: string;
  name: string;
  file: File;
}

interface VoiceSelectorProps {
  voices: string[];
  selectedVoice: string;
  onVoiceChange: (voice: string) => void;
  isZeroshotModel?: boolean;
  activeCustomPromptId?: string;  // ID of active custom prompt (empty = none selected)
  customPromptName?: string;
  uploadedPrompts?: UploadedPrompt[];
  onFileUpload?: (file: File) => void;
  onSelectPrompt?: (promptId: string) => void;
  isConfigureMode?: boolean;  // True when in configure mode (before start)
}

export function VoiceSelector({ 
  voices, 
  selectedVoice, 
  onVoiceChange,
  isZeroshotModel = false,
  activeCustomPromptId = "",
  customPromptName = "",
  uploadedPrompts = [],
  onFileUpload,
  onSelectPrompt,
  isConfigureMode = false
}: VoiceSelectorProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onFileUpload) {
      // Validate file type
      if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file');
        return;
      }
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
      }
      onFileUpload(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Check if we have any custom prompts available
  const hasBackendPrompt = customPromptName !== "";
  const hasAnyCustomPrompts = isZeroshotModel && (hasBackendPrompt || uploadedPrompts.length > 0);
  const hasActiveCustomPrompt = activeCustomPromptId !== "";
  const hasActiveDefaultVoice = selectedVoice !== "" && !hasActiveCustomPrompt;

  return (
    <div className="mt-4">
      {/* Custom Voice Prompts Section */}
      {isZeroshotModel && (
        <div className="mb-3">
          <div className="text-sm text-gray-600 mb-2">
            Custom Voices:
          </div>
          
          {/* File Upload Button - Only in Configure Mode */}
          {onFileUpload && isConfigureMode && (
            <div className="mb-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-nvidia text-white px-4 py-2 rounded-lg text-sm hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
              >
                <span>📁</span>
                Upload Custom Voice
              </button>
            </div>
          )}
          
          <select
            className={`w-full border rounded p-2 text-sm transition-opacity ${hasActiveDefaultVoice ? 'opacity-50' : 'opacity-100'}`}
            value={activeCustomPromptId}
            onChange={(e) => {
              if (e.target.value && onSelectPrompt) {
                onSelectPrompt(e.target.value);
              }
            }}
            disabled={!hasAnyCustomPrompts}
          >
            <option value="">
              {hasAnyCustomPrompts ? "Select a custom voice" : "No custom voices available"}
            </option>
            {hasBackendPrompt && (
              <option value="backend">
                {customPromptName} (Backend)
              </option>
            )}
            {uploadedPrompts.map((prompt) => (
              <option key={prompt.id} value={prompt.id}>
                {prompt.name} (Uploaded)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Default Voices Section */}
      {voices.length === 0 ? (
        <div className="text-sm text-gray-500">No voices available</div>
      ) : (
        <div>
          <div className="text-sm text-gray-600 mb-2">
            Default Voices:
          </div>
          <select
            className={`w-full border rounded p-2 text-sm transition-opacity ${hasActiveCustomPrompt ? 'opacity-50' : 'opacity-100'}`}
            value={selectedVoice}
            onChange={(e) => onVoiceChange(e.target.value)}
            onClick={(e) => {
              // Trigger handler even when clicking the already-selected option
              const target = e.target as HTMLSelectElement;
              if (target.value && target.value === selectedVoice) {
                onVoiceChange(target.value);
              }
            }}
          >
            <option value="" disabled>
              Select a voice
            </option>
            {voices.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}


