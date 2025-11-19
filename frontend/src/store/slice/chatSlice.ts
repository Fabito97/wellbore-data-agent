import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { Message, status } from "../../types";

interface ChatState {
  // All messages in the current conversation
  messages: Message[];

  // Backend conversation ID (null until first message is sent)
  conversationId: string | null;

  // Whether the current conversation is new (not persisted yet)
  isNewConversation: boolean;

  // Streaming state for assistant response
  isStreaming: boolean;
  streamingMessageId: string | null;

  // Optional buffer for streaming chunks
  streamBuffer: string;

  // Optional input draft (for UI input box)
  messageDraft: string;

  // Error state for failed requests or stream
  error: string | null;
}

const initialState: ChatState = {
  messages: [],
  conversationId: null,
  isNewConversation: true,
  isStreaming: false,
  streamingMessageId: null,
  streamBuffer: "",
  messageDraft: "",
  error: null,
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    // Add a new message (user or assistant)
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
    },

    setMessages: (state, action: PayloadAction<Message[]>) => {
      state.messages = action.payload;
    },
    // Start streaming assistant response
    startStreaming: (state, action: PayloadAction<string>) => {
      state.isStreaming = true;
      state.streamingMessageId = action.payload;
      state.streamBuffer = "";
      state.messages.push({
        id: action.payload,
        sender: "user",
        content: "",
        timestamp: Date.now(),
        isStreaming: true,
      });
    },

    updateMessageContent: (
      state,
      action: PayloadAction<{
        id: string;
        content: string;
        status: status;
        timestamp?: number;
        conversationId?: string;
      }>
    ) => {
      const msg = state.messages.find((m) => m.id === action.payload.id);
      if (msg) {
        msg.content = action.payload.content;
        msg.status = action.payload.status;
        if (action.payload.timestamp) msg.timestamp = action.payload.timestamp;
        if (action.payload.conversationId)
          msg.conversationId = action.payload.conversationId;
      }
    },

    // Append a token chunk to the streaming message
    appendStreamingContent: (
      state,
      action: PayloadAction<{ id: string; token: string }>
    ) => {
      const message = state.messages.find((m) => m.id === action.payload.id);
      if (message) {
        message.content += action.payload.token;
      }
    },

    // Stop streaming and finalize the assistant message
    stopStreaming: (state) => {
      state.isStreaming = false;
      const message = state.messages.find(
        (m) => m.id === state.streamingMessageId
      );
      if (message) {
        message.isStreaming = false;
      }
      state.streamingMessageId = null;
      state.streamBuffer = "";
    },

    // Set the conversation ID after first message is sent
    setConversationId: (state, action: PayloadAction<string>) => {
      state.conversationId = action.payload;
      state.isNewConversation = false;
    },

    // Clear all messages (e.g. when starting fresh)
    clearMessages: (state) => {
      state.messages = [];
    },

    // Set error state
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    // Set input draft (optional for UI)
    setMessageDraft: (state, action: PayloadAction<string>) => {
      state.messageDraft = action.payload;
    },

    removeMessage: (state, action: PayloadAction<string>) => {
      state.messages = state.messages.filter((m) => m.id !== action.payload);
    },

    updateMessageStatus: (
      state,
      action: PayloadAction<{ id: string; status: status }>
    ) => {
      const msg = state.messages.find((m) => m.id === action.payload.id);
      if (msg) msg.status = action.payload.status;
    },

    // appendStreamingContent: (
    //   state,
    //   action: PayloadAction<{ id: string; contentChunk: string }>
    // ) => {
    //   const msg = state.messages.find((m) => m.id === action.payload.id);
    //   if (msg) msg.content += action.payload.contentChunk;
    // },

    // Reset entire chat state (e.g. when starting new chat)
    resetChatState: (state) => {
      state.messages = [];
      state.conversationId = null;
      state.isNewConversation = true;
      state.isStreaming = false;
      state.streamingMessageId = null;
      state.streamBuffer = "";
      state.messageDraft = "";
      state.error = null;
    },
  },
});

export const {
  addMessage,
  startStreaming,
  appendStreamingContent,
  updateMessageStatus,
  stopStreaming,
  setConversationId,
  updateMessageContent,
  clearMessages,
  setMessages,
  setError,
  setMessageDraft,
  resetChatState,
} = chatSlice.actions;

export default chatSlice.reducer;
