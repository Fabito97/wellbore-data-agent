// src/features/documents/documentsApi.ts
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

interface UploadResponse {
  documentId: string;
  filename: string;
  status: string;
}

interface DocumentMetadata {
  wellName?: string;
  reportType?: string;
  parameters?: any;
}

export const documentsApi = createApi({
  reducerPath: 'documentsApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:8000/api' }),
  tagTypes: ['Documents'],
  endpoints: (builder) => ({
    uploadDocument: builder.mutation<UploadResponse, FormData>({
      query: (formData) => ({
        url: '/upload',
        method: 'POST',
        body: formData,
      }),
      invalidatesTags: ['Documents'],
    }),
    
    getDocuments: builder.query<Document[], void>({
      query: () => '/documents',
      providesTags: ['Documents'],
    }),
    
    getDocumentMetadata: builder.query<DocumentMetadata, string>({
      query: (documentId) => `/documents/${documentId}/metadata`,
    }),
    
    deleteDocument: builder.mutation<void, string>({
      query: (documentId) => ({
        url: `/documents/${documentId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Documents'],
    }),
  }),
});

export const {
  useUploadDocumentMutation,
  useGetDocumentsQuery,
  useGetDocumentMetadataQuery,
  useDeleteDocumentMutation,
} = documentsApi;