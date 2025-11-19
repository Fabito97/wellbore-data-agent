import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import { formatBytes } from "../lib/utils";

interface AddDocsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface DirectoryInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  webkitdirectory?: string;
}

const AddDocumentModal: React.FC<AddDocsModalProps> = ({ isOpen, onClose }) => {
  const [files, setFiles] = useState<File[]>([]);

  const onDrop = (acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    noClick: false,
    noKeyboard: false,
    multiple: true,
    useFsAccessApi: false, // fallback for older browsers
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black opacity-30" onClick={onClose} />

      {/* Modal Content */}
      <div className="relative z-100 bg-white dark:bg-gray-900 p-6 rounded-lg w-full max-w-xl shadow-lg">
        <h2 className="text-lg font-bold mb-4 text-gray-800 dark:text-gray-100">
          Upload Documents
        </h2>

        <div
          {...getRootProps()}
          className="border-dashed border-2 border-gray-400 dark:border-gray-600 p-6 rounded-lg text-center cursor-pointer"
        >
          <input {...(getInputProps() as any)} webkitdirectory="true" />
          <p className="text-gray-600 dark:text-gray-300">
            Drag and drop files or folders here, or click to select
          </p>
        </div>

        <ul className="mt-4 space-y-2 max-h-60 overflow-y-auto flex flex-col">
          {files.map((file, index) => (
            <li
              key={index}
              className="flex justify-between items-center text-sm text-gray-700 dark:text-gray-200"
            >
              <span>{file.name.slice(0, 20)}</span>
              <span>{formatBytes(file.size)}</span>
              {/* <span>{file.type || "Unknown"}</span> */}
            </li>
          ))}
        </ul>

        <div className="mt-6 flex justify-end gap-2">
          <button onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button
            onClick={() => console.log("Upload logic here")}
            className="btn-primary"
          >
            Upload
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddDocumentModal;

