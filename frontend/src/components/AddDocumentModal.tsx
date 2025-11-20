import React, { useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { formatBytes } from "../lib/utils";
import { useUploader } from "../lib/hooks/useUploader";
import { useUploadDocumentMutation } from "../store/api/documentsApi";

interface AddDocsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const ACCEPTED_FORMATS = [
  ".pdf",
  ".csv",
  ".xlsx",
  ".xls",
  ".las",
  ".txt",
  ".docx",
  ".doc",
  ".zip",
];

const AddDocumentModal: React.FC<AddDocsModalProps> = ({ isOpen, onClose }) => {
  const { items, addFiles, addUrl, cancelUpload, removeItem, clearAll } =
    useUploader();
  const [uploadDocument] = useUploadDocumentMutation();
  const [isProcessing, setIsProcessing] = useState(false);
  const [processMessage, setProcessMessage] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const urlRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  const onDrop = (acceptedFiles: File[]) => {
    addFiles(acceptedFiles);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    noClick: false,
    noKeyboard: false,
    multiple: true,
    useFsAccessApi: false,
  });

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFolderClick = () => {
    folderInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files));
    }
  };

  const handleFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files));
    }
  };

  const recommendZip = useMemo(() => {
    return "We recommend uploading a zip file of your report to maintain to maintain structure and improve processing speed.";
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-10 flex items-center justify-center">
      <div
        className="absolute inset-0 z-10 bg-black opacity-10"
        onClick={onClose}
      />

      <div className="relative z-50 bg-white dark:bg-gray-900 p-6 rounded-lg w-full max-w-4xl shadow-2xl">
        <div className="flex items-start justify-between mb-4">
          <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100">
            Add Documents
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-300"
          >
            ✕
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
          <div onClick={handleFileClick}>
            <div
              {...getRootProps()}
              className="border-2 border-dashed border-gray-400 dark:border-gray-600 p-6 rounded-lg text-center cursor-pointer bg-gray-50 dark:bg-gray-800"
            >
              <input
                {...(getInputProps() as any)}
                ref={fileInputRef}
                onChange={handleFileChange}
              />
              <div className="flex flex-col items-center gap-2">
                <div className="text-3xl">📁</div>
                <p className="text-gray-600 dark:text-gray-300">
                  Drag & Drop or Choose file to upload
                </p>
                <small className="text-xs text-gray-400">
                  {ACCEPTED_FORMATS.join(", ")}
                </small>
              </div>
            </div>

            {items.length === 0 && (
              <div className="mt-4">
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={handleFileClick}
                    className="flex-1 px-3 py-2 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Choose Files
                  </button>
                  <button
                    onClick={handleFolderClick}
                    className="flex-1 px-3 py-2 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Choose Folder
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
                <input
                  ref={folderInputRef}
                  type="file"
                  multiple
                  onChange={handleFolderChange}
                  style={{ display: "none" }}
                  {...({ webkitdirectory: "true" } as any)}
                />

                <div className="text-xs text-center text-gray-500 dark:text-gray-400 mt-3">
                  {recommendZip}
                </div>
                <div className="text-xs text-center text-gray-500 dark:text-gray-400 mt-3">
                  -------------------or--------------------
                </div>
                <label className="block text-sm text-gray-500 dark:text-gray-400 mb-2 mt-3">
                  Import from URL
                </label>
                <div className="flex gap-2 px-3 py-2 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700">
                  <input
                    ref={urlRef}
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="Add file URL"
                    className="flex-1 text-sm"
                  />
                  <button
                    disabled={!urlInput}
                    onClick={() => {
                      if (!urlInput) return;
                      addUrl(urlInput);
                      setUrlInput("");
                    }}
                    className={`btn- px-4 py-2 ${
                      !urlInput && "opacity-50 cursor-disabled"
                    } hover:text-gray-300`}
                  >
                    Upload
                  </button>
                </div>
              </div>
            )}
          </div>

          <div>
            {items.length === 0 ? (
              <div className="text-sm text-center text-gray-500">
                No files queued. Add files to begin.
              </div>
            ) : (
              <>
                <h2 className="mb-5">File preview</h2>
                <div className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700 max-h-[260px] overflow-y-auto">
                  {items.map((it) => (
                    <div key={it.id} className="mb-3">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded flex items-center justify-center text-xs">
                            {it.name.split(".").pop()?.toUpperCase() || "F"}
                          </div>
                          <div>
                            <div className="text-sm font-medium text-gray-800 dark:text-gray-100">
                              {it.name}
                            </div>
                            <div className="text-xs text-gray-400">
                              {it.size ? formatBytes(it.size) : it.url || ""}
                            </div>
                          </div>
                        </div>

                        <div className="text-right text-xs">
                          <div>
                            {it.status === "uploading"
                              ? `${it.progress}%`
                              : it.status}
                          </div>
                        </div>
                      </div>

                      <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                        <div
                          style={{ width: `${it.progress}%` }}
                          className={`h-2 bg-emerald-500 transition-all ${
                            it.status === "completed" ? "bg-green-500" : ""
                          }`}
                        />
                      </div>

                      <div className="flex gap-2 mt-2">
                        {it.status === "uploading" && (
                          <button
                            onClick={() => cancelUpload(it.id)}
                            className="px-3 py-1 text-xs rounded bg-gray-100 dark:bg-gray-700"
                          >
                            Cancel
                          </button>
                        )}
                        {it.status === "error" && (
                          <button
                            onClick={() => {
                              /* retry by re-queueing */
                            }}
                            className="px-3 py-1 text-xs rounded bg-yellow-500 text-white"
                          >
                            Retry
                          </button>
                        )}
                        <button
                          onClick={() => removeItem(it.id)}
                          className="px-3 py-1 text-xs rounded bg-gray-100 dark:bg-gray-700"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        <div className="mt-6 flex justify-between items-center">
          <div className="text-xs text-gray-400">
            Need help? Visit our Help Center
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => {
                clearAll();
                onClose();
              }}
              className="hover:bg-gray-600 text-gray-500 hover:text-white px-3 rounded-md"
              disabled={isProcessing}
            >
              Cancel
            </button>
            <button
              disabled={items.length === 0 || isProcessing}
              onClick={async () => {
                // Perform backend uploads (sequential for now)
                if (items.length === 0) return;
                setIsProcessing(true);
                setProcessMessage("Uploading documents...");
                try {
                  for (const it of items) {
                    // build FormData per item
                    const form = new FormData();
                    if (it.file) {
                      form.append("file", it.file as File, it.name);
                    } else if (it.url) {
                      form.append("url", it.url);
                      // optionally include filename
                      form.append("filename", it.name);
                    }

                    // call mutation and await response
                    await uploadDocument(form).unwrap();
                  }

                  setProcessMessage("Processing completed.");
                  // small delay so user sees success
                  setTimeout(() => {
                    clearAll();
                    setIsProcessing(false);
                    setProcessMessage(null);
                    onClose();
                  }, 900);
                } catch (err) {
                  console.error("Upload failed:", err);
                  setProcessMessage(
                    "Upload failed. Please try again or check the file."
                  );
                  setIsProcessing(false);
                }
              }}
              className={`btn ${
                (items.length === 0 || isProcessing) &&
                "opacity-50 cursor-disabled"
              } p-2 rounded-md bg-blue-700 hover:bg-blue-800`}
            >
              {isProcessing ? "Processing..." : "Process"}
            </button>
          </div>
        </div>

        {isProcessing && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
            <div className="bg-white dark:bg-gray-900 p-6 rounded-lg flex flex-col items-center gap-3 shadow">
              <div className="w-10 h-10 rounded-full border-4 border-t-transparent border-blue-600 animate-spin" />
              <div className="text-sm text-gray-800 dark:text-gray-100">
                {processMessage}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AddDocumentModal;
