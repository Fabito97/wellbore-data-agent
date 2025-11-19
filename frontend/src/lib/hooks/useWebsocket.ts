import { useEffect, useRef, useState, useCallback } from "react";

interface WSMessage {
  type: string;
  [key: string]: any;
}

interface WebSocketHookOptions {
  onConnected?: () => void;
  onStatus?: (msg: string) => void;
  onAnswer?: (data: any) => void;
  onSummary?: (data: any) => void;
  onTables?: (data: any) => void;
  onError?: (msg: string) => void;
}

const WS_URL = "ws://localhost:8000/api/v1/ws";

export function useWebSocketChat(options: WebSocketHookOptions = {}) {
  const {
    onConnected,
    onStatus,
    onAnswer,
    onSummary,
    onTables,
    onError,
  } = options;

  const socketRef = useRef<WebSocket | null>(null);
  const retryTimeout = useRef<number | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const messageQueue = useRef<any[]>([]);

  const connect = useCallback(() => {
    if (
      socketRef.current &&
      (socketRef.current.readyState === WebSocket.OPEN ||
        socketRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const ws = new WebSocket(WS_URL);
    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);

      if (onConnected) onConnected();

      // Flush queued messages
      messageQueue.current.forEach((msg) =>
        ws.send(JSON.stringify(msg))
      );
      messageQueue.current = [];
    };

    ws.onmessage = (event) => {
      let msg: WSMessage;

      try {
        msg = JSON.parse(event.data);
      } catch {
        console.warn("Invalid WS message:", event.data);
        return;
      }

      switch (msg.type) {
        case "connected":
          onStatus?.("Connected to Wellbore AI Agent");
          break;

        case "status":
          onStatus?.(msg.message);
          break;

        case "answer":
          onAnswer?.(msg);
          break;

        case "summary":
          onSummary?.(msg);
          break;

        case "tables":
          onTables?.(msg);
          break;

        case "error":
          onError?.(msg.message);
          break;

        default:
          console.warn("Unknown WS message:", msg);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);

      // Autoreconnect after 1.5s
      retryTimeout.current = setTimeout(() => {
        connect();
      }, 1500);
    };

    ws.onerror = () => {
      onError?.("WebSocket encountered an error.");
      ws.close();
    };
  }, [onConnected, onStatus, onAnswer, onSummary, onTables, onError]);

  useEffect(() => {
    connect();
    return () => {
      retryTimeout.current && clearTimeout(retryTimeout.current);
      socketRef.current?.close();
    };
  }, [connect]);

  const sendMessage = (data: any) => {
    const ws = socketRef.current;

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // Queue until connected
      messageQueue.current.push(data);
      return;
    }

    ws.send(JSON.stringify(data));
  };

  // -----------------------------
  // PUBLIC API
  // -----------------------------
  const askQuestion = (content: string, options: any = {}) => {
    sendMessage({
      type: "question",
      content,
      options,
    });
  };

  const summarizeDocument = (document_id: string, max_words = 200) => {
    sendMessage({
      type: "summarize",
      document_id,
      max_words,
    });
  };

  const extractTables = (query: string, top_k = 3) => {
    sendMessage({
      type: "extract_tables",
      query,
      top_k,
    });
  };

  return {
    isConnected,
    askQuestion,
    summarizeDocument,
    extractTables,
  };
}
