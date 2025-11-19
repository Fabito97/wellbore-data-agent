// src/services/websocket.ts
import { store } from "../store/store";
import {
  startStreaming,
  appendStreamingContent,
  stopStreaming,
  setError,
} from "../store/slice/chatSlice";
import { v4 as uuidv4 } from "uuid";

class WebSocketService {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(sessionId: string) {
    this.sessionId = sessionId;
    const wsUrl = `ws://localhost:8000/ws/chat/${sessionId}`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("WebSocket connected");
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      store.dispatch(setError("Connection error occurred"));
    };

    this.ws.onclose = () => {
      console.log("WebSocket disconnected");
      this.reconnect();
    };
  }

  private handleMessage(data: any) {
    switch (data.type) {
      case "start":
        store.dispatch(startStreaming(data.messageId || uuidv4()));
        break;

      case "token":
        const messageId = store.getState().chat.streamingMessageId;
        if (messageId) {
          store.dispatch(
            appendStreamingContent({
              id: messageId,
              token: data.token,
            })
          );
        }
        break;

      case "visualization":
        // Handle visualization data
        // Can dispatch to analysis slice
        break;

      case "complete":
        store.dispatch(stopStreaming());
        break;

      case "error":
        store.dispatch(setError(data.message));
        store.dispatch(stopStreaming());
        break;
    }
  }

  sendMessage(message: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ message }));
    } else {
      store.dispatch(setError("Not connected to server"));
    }
  }

  private reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.sessionId) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
      setTimeout(() => this.connect(this.sessionId!), 2000);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const wsService = new WebSocketService();
