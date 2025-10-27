// src/features/chat/chatSlice.ts
import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  metadata?: {
    sources?: string[];
    confidence?: number;
  };
}

interface ChatState {
  messages: Message[];
  currentSessionId: string | null;
  isStreaming: boolean;
  streamingMessageId: string | null;
  error: string | null;
}

const initialState: ChatState = {
  messages: [],
  currentSessionId: null,
  isStreaming: false,
  streamingMessageId: null,
  error: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },
    
    startStreaming: (state, action: PayloadAction<string>) => {
      state.isStreaming = true;
      state.streamingMessageId = action.payload;
      state.messages.push({
        id: action.payload,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        isStreaming: true,
      });
    },
    
    appendStreamingContent: (state, action: PayloadAction<{ id: string; token: string }>) => {
      const message = state.messages.find(m => m.id === action.payload.id);
      if (message) {
        message.content += action.payload.token;
      }
    },
    
    stopStreaming: (state) => {
      state.isStreaming = false;
      const message = state.messages.find(m => m.id === state.streamingMessageId);
      if (message) {
        message.isStreaming = false;
      }
      state.streamingMessageId = null;
    },
    
    setSessionId: (state, action: PayloadAction<string>) => {
      state.currentSessionId = action.payload;
    },
    
    clearMessages: (state) => {
      state.messages = [];
    },
    
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  addMessage,
  startStreaming,
  appendStreamingContent,
  stopStreaming,
  setSessionId,
  clearMessages,
  setError,
} = chatSlice.actions;

export default chatSlice.reducer;