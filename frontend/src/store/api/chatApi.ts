// src/features/chat/chatApi.ts
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

interface ChatRequest {
  message: string;
  sessionId: string;
}

interface ChatResponse {
  response: string;
  sessionId: string;
  metadata?: any;
}

interface ChatHistory {
  messages: Array<{
    role: string;
    content: string;
    timestamp: number;
  }>;
}

export const chatApi = createApi({
  reducerPath: 'chatApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:8000/api' }),
  tagTypes: ['ChatHistory'],
  endpoints: (builder) => ({
    sendMessage: builder.mutation<ChatResponse, ChatRequest>({
      query: (body) => ({
        url: '/chat',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['ChatHistory'],
    }),
    
    getChatHistory: builder.query<ChatHistory, string>({
      query: (sessionId) => `/sessions/${sessionId}/history`,
      providesTags: ['ChatHistory'],
    }),
    
    createSession: builder.mutation<{ sessionId: string }, void>({
      query: () => ({
        url: '/sessions',
        method: 'POST',
      }),
    }),
  }),
});

export const {
  useSendMessageMutation,
  useGetChatHistoryQuery,
  useCreateSessionMutation,
} = chatApi;