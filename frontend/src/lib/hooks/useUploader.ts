import { useCallback, useEffect, useRef, useState } from "react";

export type UploadStatus =
  | "queued"
  | "uploading"
  | "completed"
  | "error"
  | "canceled";

export interface UploadItem {
  id: string;
  name: string;
  size?: number;
  type?: string;
  progress: number; // 0-100
  status: UploadStatus;
  file?: File;
  url?: string;
  error?: string | null;
}

// Simulate an upload to backend; replace with real API call later.
function simulateUpload(item: UploadItem, onProgress: (p: number) => void) {
  return new Promise<void>((resolve, reject) => {
    let progress = 0;
    const interval = 120 + Math.floor(Math.random() * 200);
    const id = setInterval(() => {
      progress += Math.floor(Math.random() * 12) + 4;
      if (progress >= 100) {
        onProgress(100);
        clearInterval(id);
        // small delay to simulate processing
        setTimeout(() => resolve(), 400 + Math.floor(Math.random() * 800));
        return;
      }
      onProgress(Math.min(100, progress));
    }, interval);
    // allow cancellation by returning clearInterval handle via closure
  });
}

export function useUploader() {
  const [items, setItems] = useState<UploadItem[]>([]);
  const uploadingRef = useRef<Record<string, boolean>>({});

  const addFiles = useCallback((files: File[]) => {
    const newItems: UploadItem[] = files.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
      size: f.size,
      type: f.type,
      progress: 0,
      status: "queued",
      file: f,
      error: null,
    }));
    setItems((s) => [...s, ...newItems]);
  }, []);

  const addUrl = useCallback((url: string) => {
    const name = url.split("/").pop() || url;
    const newItem: UploadItem = {
      id: crypto.randomUUID(),
      name,
      progress: 0,
      status: "queued",
      url,
      error: null,
    };
    setItems((s) => [...s, newItem]);
  }, []);

  const startUpload = useCallback(
    async (itemId?: string) => {
      const toStart = itemId
        ? items.filter((i) => i.id === itemId && i.status === "queued")
        : items.filter((i) => i.status === "queued");

      for (const item of toStart) {
        if (uploadingRef.current[item.id]) continue;
        uploadingRef.current[item.id] = true;
        setItems((s) =>
          s.map((it) =>
            it.id === item.id ? { ...it, status: "uploading", progress: 0 } : it
          )
        );
        try {
          await simulateUpload(item, (p) => {
            setItems((s) =>
              s.map((it) => (it.id === item.id ? { ...it, progress: p } : it))
            );
          });
          // mark completed with fake backend response shape
          setItems((s) =>
            s.map((it) =>
              it.id === item.id
                ? { ...it, progress: 100, status: "completed" }
                : it
            )
          );
        } catch (err) {
          setItems((s) =>
            s.map((it) =>
              it.id === item.id
                ? {
                    ...it,
                    status: "error",
                    error: (err as any)?.message || "Upload failed",
                  }
                : it
            )
          );
        } finally {
          uploadingRef.current[item.id] = false;
        }
      }
    },
    [items]
  );

  const cancelUpload = useCallback((id: string) => {
    // Since simulation uses setInterval inside simulateUpload which isn't exposed for cancellation,
    // we simply mark canceled. In real implementation, cancel a fetch/XHR request or abort controller.
    setItems((s) =>
      s.map((it) =>
        it.id === id
          ? { ...it, status: "canceled", error: "Canceled by user" }
          : it
      )
    );
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((s) => s.filter((it) => it.id !== id));
  }, []);

  const clearAll = useCallback(() => setItems([]), []);

  useEffect(() => {
    // Auto-start queued uploads (limit parallelism if desired)
    const queued = items.filter((i) => i.status === "queued");
    if (queued.length === 0) return;
    const running = items.filter((i) => i.status === "uploading").length;
    const parallel = 2; // limit concurrent uploads
    const available = Math.max(0, parallel - running);
    if (available <= 0) return;
    const toStart = queued.slice(0, available);
    toStart.forEach((it) => startUpload(it.id));
  }, [items, startUpload]);

  return {
    items,
    addFiles,
    addUrl,
    startUpload,
    cancelUpload,
    removeItem,
    clearAll,
  } as const;
}
