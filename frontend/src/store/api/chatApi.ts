// features/chat/chatApi.ts
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { Conversation, Message } from '../../types';

interface MessageRequest {
  status: 'pending' | 'sent' | 'error' | 'streaming'
  data: Message
}

const BaseUrl = 'http://127.0.0.1:8000'
export const chatApi = createApi({
  reducerPath: 'chatApi',
  baseQuery: fetchBaseQuery({ baseUrl: `${BaseUrl}/api/v1` }),
  tagTypes: ['Conversations', 'Messages'],
  endpoints: (builder) => ({
    getConversations: builder.query<Conversation[], void>({
      query: () => '/chat',
      providesTags: ['Conversations'],
    }),
    getMessages: builder.query<Message[], string>({
      query: (conversationId) => `/chat/${conversationId}/messages`,
      providesTags: (result, error, id) => [{ type: 'Messages', id }],
    }),
    sendMessage: builder.mutation<MessageRequest, { conversationId?: string; question: string }>({
      query: ({ conversationId, question }) => ({
        url: '/chat/ask',
        method: 'POST',
        body: { conversation_id: conversationId, query: question },
      }),
      invalidatesTags: ['Conversations'],
    }),
    streamMessage: builder.mutation<any, { conversationId?: string; question: string }>({
      query: ({ conversationId, question }) => ({
        url: '/chat/stream',
        method: 'POST',
        body: { conversation_id: conversationId, question },
      }),
    }),
    deleteConversation: builder.mutation<void, string>({
      query: (conversationId) => ({
        url: `/chat/${conversationId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Conversations'],
    }),
  }),
});

export const {
  useGetConversationsQuery,
  useGetMessagesQuery,
  useLazyGetMessagesQuery,
  useSendMessageMutation,
  useStreamMessageMutation,
  useDeleteConversationMutation,
} = chatApi;